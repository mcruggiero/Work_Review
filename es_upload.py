# ========================================================================
# SCORECARD ES UPLOAD MODULE (Standalone)
# Uploads ONLY the latest note per SID to Elasticsearch with GPT enrichment
# Uses ScoreCardConfig for connection info only - no pipeline dependencies
# ========================================================================

import os
import time
import json
from pathlib import Path
from typing import Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyodbc
import torch
import tiktoken
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from .config import ScoreCardConfig


# ========================================================================
# SQL QUERY FOR LATEST NOTES
# ========================================================================

LATEST_NOTES_QUERY = """
WITH RankedNotes AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY SID
            ORDER BY Note_Year DESC, Note_Month DESC, Scorecard_Detail_Note_SID DESC
        ) AS rn
    FROM (
        SELECT
            sd.SID,
            sd.LM_Vendor_ID,
            sd.Program_Name,
            sdn.Scorecard_Detail_Note_SID,
            sdn.Note_Year,
            sdn.Note_Month,
            sdn.Scorecard_Note,
            sdn.Overall,
            sdn.Cost,
            sdn.Schedule,
            sdn.Quality,
            sdn.Responsiveness,
            sdn.Report_Year,
            sdn.Report_Month
        FROM dbo.Scorecard_Detail sd
        INNER JOIN dbo.Scorecard_Detail_Note sdn
            ON sd.Scorecard_Detail_SID = sdn.Scorecard_Detail_SID
        WHERE sdn.Scorecard_Note IS NOT NULL
            AND LEN(sdn.Scorecard_Note) > 10
    ) sub
)
SELECT
    SID,
    LM_Vendor_ID,
    Program_Name,
    Scorecard_Detail_Note_SID,
    Note_Year,
    Note_Month,
    Scorecard_Note,
    Overall,
    Cost,
    Schedule,
    Quality,
    Responsiveness,
    Report_Year,
    Report_Month,
    CONCAT(
        RIGHT('000000' + CAST(SID AS VARCHAR), 6), '.',
        CAST(Note_Year AS VARCHAR), '.',
        RIGHT('00' + CAST(Note_Month AS VARCHAR), 2), '.',
        RIGHT('000000' + CAST(Scorecard_Detail_Note_SID AS VARCHAR), 6)
    ) AS sid_key
FROM RankedNotes
WHERE rn = 1
ORDER BY SID
"""


# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def _report(tag: str, msg: str) -> None:
    """Print a tagged status message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{tag}] {msg}")


def _clean_for_json(val: Any) -> Any:
    """Convert value to JSON-serializable format."""
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return [_clean_for_json(v) for v in val]
    return val


def _serialize_row(row: pd.Series, drop_keys: set = None) -> dict:
    """Serialize a DataFrame row to ES-compatible dict."""
    drop_keys = drop_keys or {"_id"}
    result = {}
    for k, v in row.items():
        if k in drop_keys:
            continue
        result[k] = _clean_for_json(v)
    return result


# ========================================================================
# MAIN UPLOADER CLASS
# ========================================================================

class StandaloneESUploader:
    """
    Standalone ES uploader - uses ScoreCardConfig for connections only.

    No dependency on ScoreCardState or the main pipeline.

    Usage:
        config = ScoreCardConfig()
        uploader = StandaloneESUploader(config)
        stats = uploader.run(output_csv="uploaded_notes.csv")
    """

    def __init__(self, config: ScoreCardConfig) -> None:
        self.config = config
        self.sql_conn = None
        self.es_client = None
        self.gpt_client = None
        self.embedding_model = None
        self.gpt_prompt = None

    def _connect_sql(self) -> None:
        """Establish SQL Server connection."""
        _report("SQL", f"Connecting to {self.config.sql_server}...")

        conn_str = (
            f"DRIVER={{{self.config.sql_driver_path}}};"
            f"SERVER={self.config.sql_server};"
            f"DATABASE={self.config.sql_database};"
            f"UID={self.config.sql_uid};"
            f"PWD={self.config.sql_pwd}"
        )

        self.sql_conn = pyodbc.connect(conn_str)
        _report("SQL", "Connected!")

    def _connect_es(self) -> None:
        """Establish Elasticsearch connection."""
        _report("ES", f"Connecting to {self.config.es_host}...")

        self.es_client = Elasticsearch(self.config.es_host)
        if not self.es_client.ping():
            raise ConnectionError("Elasticsearch ping failed")

        _report("ES", "Connected!")

    def _connect_gpt(self) -> None:
        """Initialize GPT client (optional)."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            _report("GPT", "OPENAI_API_KEY not set - skipping GPT enrichment")
            return

        _report("GPT", f"Initializing GPT client: {self.config.gpt_base_url}")
        self.gpt_client = OpenAI(
            base_url=self.config.gpt_base_url,
            api_key=api_key
        )
        _report("GPT", "GPT client ready!")

    def _load_embedding_model(self) -> None:
        """Load the sentence embedding model."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _report("EMBD", f"Loading {self.config.embedding_model_name} on {device}...")

        start = time.time()
        self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
        self.embedding_model = self.embedding_model.to(device)
        elapsed = time.time() - start

        _report("EMBD", f"Model loaded in {elapsed:.1f}s")

    def _load_gpt_prompt(self) -> None:
        """Load GPT prompt template from config path."""
        prompt_path = self.config.gpt_prompt_location
        if prompt_path and os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.gpt_prompt = f.read()
            _report("GPT", f"Loaded prompt from {prompt_path}")
        else:
            # Default simple prompt
            self.gpt_prompt = """
You are analyzing a subcontract scorecard note. Based on the note content and historical patterns,
provide a brief justification for the predicted outcome.

SID: {SID}
Vendor: {LM_Vendor_ID}
Program: {Program_Name}
Current Overall Rating: {Overall}

Note Content:
{Scorecard_Note}

Provide a 2-3 sentence analysis of this subcontract's status and any concerns.
"""
            _report("GPT", "Using default prompt template")

    def _query_latest_notes(self) -> pd.DataFrame:
        """Query SQL for latest note per SID."""
        _report("SQL", "Querying latest notes per SID...")

        df = pd.read_sql(LATEST_NOTES_QUERY, self.sql_conn)
        _report("SQL", f"Retrieved {len(df)} latest notes")

        return df

    def _generate_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate embeddings for all notes."""
        df = df.copy()
        df["text_for_embedding"] = df["Scorecard_Note"].fillna("").astype(str)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _report("EMBD", f"Encoding {len(df)} notes...")

        embeddings = self.embedding_model.encode(
            df["text_for_embedding"].tolist(),
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            device=device
        )
        df["embedding"] = embeddings.tolist()

        # Token counts
        tokenizer = tiktoken.get_encoding(self.config.tokenizer_name)
        df["token_count"] = df["text_for_embedding"].apply(
            lambda t: len(tokenizer.encode(t))
        )

        return df

    def _generate_single_justification(self, row: pd.Series) -> str:
        """Generate GPT justification for a single row."""
        prompt = self.gpt_prompt.format(
            SID=row.get("SID", "Unknown"),
            LM_Vendor_ID=row.get("LM_Vendor_ID", "Unknown"),
            Program_Name=row.get("Program_Name", "Unknown"),
            Overall=row.get("Overall", "Unknown"),
            Scorecard_Note=row.get("Scorecard_Note", "")[:2000]  # Truncate if too long
        )

        response = self.gpt_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    def _generate_justifications(
        self,
        df: pd.DataFrame,
        max_workers: int = 4,
        max_attempts: int = 3
    ) -> pd.DataFrame:
        """Generate GPT justifications for all rows."""
        if self.gpt_client is None:
            _report("GPT", "No GPT client - skipping justifications")
            df["gpt_justification"] = ""
            return df

        df = df.copy()
        _report("GPT", f"Generating justifications for {len(df)} notes...")

        results = {}

        def process_row(idx: int, row: pd.Series):
            sid = row["SID"]
            for attempt in range(1, max_attempts + 1):
                try:
                    if attempt > 1:
                        time.sleep(attempt * 1.5)
                    justification = self._generate_single_justification(row)
                    return sid, justification
                except Exception as e:
                    if attempt == max_attempts:
                        _report("GPT", f"SID {sid} failed after {max_attempts} attempts: {e}")
                        return sid, f"[Error: {str(e)[:100]}]"

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_row, i, row): row["SID"]
                for i, row in df.iterrows()
            }

            completed = 0
            for future in as_completed(futures):
                sid, justification = future.result()
                results[sid] = justification
                completed += 1
                if completed % 10 == 0:
                    _report("GPT", f"Progress: {completed}/{len(df)}")

        df["gpt_justification"] = df["SID"].map(results)

        success = sum(1 for v in results.values() if not v.startswith("[Error"))
        _report("GPT", f"Justifications complete: {success}/{len(df)} succeeded")

        return df

    def _create_es_index(self, index_name: str) -> None:
        """Create or recreate the ES index with proper mapping."""
        if self.es_client.indices.exists(index=index_name):
            self.es_client.indices.delete(index=index_name)
            _report("ES", f"Deleted existing index '{index_name}'")

        mapping = {
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.config.embedding_vector_dim,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "token_count": {"type": "integer"},
                    "embedding_version": {"type": "keyword"},
                    "gpt_justification": {"type": "text"},
                    "created_at": {"type": "date"},
                    "SID": {"type": "integer"},
                    "LM_Vendor_ID": {"type": "keyword"},
                    "Program_Name": {"type": "text"},
                    "Note_Year": {"type": "integer"},
                    "Note_Month": {"type": "integer"},
                    "Overall": {"type": "keyword"},
                    "Cost": {"type": "keyword"},
                    "Schedule": {"type": "keyword"},
                    "Quality": {"type": "keyword"},
                    "Responsiveness": {"type": "keyword"},
                    "Scorecard_Note": {"type": "text"},
                    "sid_key": {"type": "keyword"},
                }
            }
        }

        self.es_client.indices.create(index=index_name, body=mapping)
        _report("ES", f"Created index '{index_name}'")

    def _bulk_index(self, df: pd.DataFrame, index_name: str) -> tuple[int, list]:
        """Bulk index documents to Elasticsearch."""
        # Add metadata
        df = df.copy()
        df["embedding_version"] = self.config.embedding_model_name
        df["created_at"] = datetime.utcnow().isoformat()
        df["_id"] = df["sid_key"].astype(str)

        actions = []
        for _, row in df.iterrows():
            actions.append({
                "_index": index_name,
                "_id": row["_id"],
                "_source": _serialize_row(row, drop_keys={"_id", "text_for_embedding"})
            })

        success, errors = bulk(self.es_client, actions, raise_on_error=False)
        return success, errors

    def _save_csv(self, df: pd.DataFrame, output_csv: str) -> str:
        """Save uploaded data to CSV."""
        # Columns to save (exclude embedding - too large)
        csv_cols = [
            "SID", "sid_key", "LM_Vendor_ID", "Program_Name",
            "Note_Year", "Note_Month", "Scorecard_Detail_Note_SID",
            "Overall", "Cost", "Schedule", "Quality", "Responsiveness",
            "Scorecard_Note", "gpt_justification", "token_count",
            "created_at", "embedding_version"
        ]

        # Only include columns that exist
        csv_cols = [c for c in csv_cols if c in df.columns]

        # Ensure directory exists
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df[csv_cols].to_csv(output_path, index=False)
        _report("CSV", f"Saved {len(df)} rows to {output_path}")

        return str(output_path)

    def run(
        self,
        index_name: Optional[str] = None,
        output_csv: str = "es_upload_output.csv",
        generate_justifications: bool = True,
        limit: Optional[int] = None
    ) -> dict:
        """
        Run the full upload pipeline.

        Args:
            index_name: ES index name (default: config.rag_index + "_latest")
            output_csv: Path for output CSV file
            generate_justifications: Whether to generate GPT justifications
            limit: Limit records for testing

        Returns:
            Dict with upload statistics
        """
        # Default index name
        if index_name is None:
            index_name = f"{self.config.rag_index}_latest"

        stats = {
            "started_at": datetime.utcnow().isoformat(),
            "total_notes": 0,
            "indexed": 0,
            "errors": 0,
            "index_name": index_name,
            "csv_path": None
        }

        _report("START", "=" * 60)
        _report("START", "Standalone ES Upload: Latest Notes with GPT Enrichment")
        _report("START", "=" * 60)

        try:
            # Step 1: Connect to everything
            self._connect_sql()
            self._connect_es()
            if generate_justifications:
                self._connect_gpt()
                self._load_gpt_prompt()
            self._load_embedding_model()

            # Step 2: Query latest notes from SQL
            df = self._query_latest_notes()
            stats["total_notes"] = len(df)

            if limit:
                df = df.head(limit)
                _report("DATA", f"Limited to {len(df)} records for testing")

            # Step 3: Generate embeddings
            df = self._generate_embeddings(df)

            # Step 4: Generate GPT justifications
            if generate_justifications:
                df = self._generate_justifications(df)
            else:
                df["gpt_justification"] = ""

            # Step 5: Add metadata
            df["created_at"] = datetime.utcnow().isoformat()
            df["embedding_version"] = self.config.embedding_model_name

            # Step 6: Create ES index and upload
            self._create_es_index(index_name)
            success, errors = self._bulk_index(df, index_name)

            stats["indexed"] = success
            stats["errors"] = len(errors) if errors else 0

            _report("ES", f"Indexed {success} documents to '{index_name}'")

            if errors:
                _report("ES", f"{len(errors)} documents failed to index")
                for err in errors[:3]:
                    print(json.dumps(err, indent=2))

            # Step 7: Save CSV
            csv_path = self._save_csv(df, output_csv)
            stats["csv_path"] = csv_path

        except Exception as e:
            stats["error"] = str(e)
            _report("ERROR", f"Pipeline failed: {e}")
            raise

        finally:
            # Cleanup
            if self.sql_conn:
                self.sql_conn.close()

            stats["completed_at"] = datetime.utcnow().isoformat()

        _report("DONE", "=" * 60)
        _report("DONE", f"Upload complete: {stats['indexed']}/{stats['total_notes']} indexed")
        _report("DONE", f"CSV output: {stats['csv_path']}")
        _report("DONE", "=" * 60)

        return stats


# ========================================================================
# CONVENIENCE FUNCTION
# ========================================================================

def upload_latest_to_es(
    config: Optional[ScoreCardConfig] = None,
    index_name: Optional[str] = None,
    output_csv: str = "es_upload_output.csv",
    generate_justifications: bool = True,
    limit: Optional[int] = None
) -> dict:
    """
    Convenience function to upload latest notes per SID to Elasticsearch.

    Uses ScoreCardConfig for all connection settings.

    Args:
        config: ScoreCardConfig instance (creates default if None)
        index_name: ES index name (default: rag_index + "_latest")
        output_csv: Path for output CSV file
        generate_justifications: Whether to generate GPT justifications
        limit: Limit records for testing

    Returns:
        Dict with upload statistics including csv_path

    Example:
        from scorecard import ScoreCardConfig, upload_latest_to_es

        config = ScoreCardConfig()
        stats = upload_latest_to_es(
            config=config,
            output_csv="uploaded_notes.csv",
            generate_justifications=True,
            limit=10  # for testing
        )
        print(f"Uploaded {stats['indexed']} notes")
        print(f"CSV saved to: {stats['csv_path']}")
    """
    if config is None:
        config = ScoreCardConfig()

    uploader = StandaloneESUploader(config)
    return uploader.run(
        index_name=index_name,
        output_csv=output_csv,
        generate_justifications=generate_justifications,
        limit=limit
    )


# ========================================================================
# CLI ENTRY POINT
# ========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload latest scorecard notes to Elasticsearch"
    )
    parser.add_argument("--es-index", help="ES index name (default: from config)")
    parser.add_argument("--output-csv", default="es_upload_output.csv")
    parser.add_argument("--no-gpt", action="store_true", help="Skip GPT justifications")
    parser.add_argument("--limit", type=int, help="Limit records for testing")

    args = parser.parse_args()

    # Uses ScoreCardConfig for all connection info
    config = ScoreCardConfig()

    stats = upload_latest_to_es(
        config=config,
        index_name=args.es_index,
        output_csv=args.output_csv,
        generate_justifications=not args.no_gpt,
        limit=args.limit
    )

    print(f"\nFinal stats: {json.dumps(stats, indent=2)}")
