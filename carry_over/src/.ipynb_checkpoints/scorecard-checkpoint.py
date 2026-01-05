# ========================================================================
# CORE PYTHON LIBRARIES
# ========================================================================
import os
import re
import gc
import ast
import json
import warnings
from typing import Any, Literal, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import time
from collections import defaultdict

# =============================================================
# DATA HANDLING + NUMPY
# =============================================================
import numpy as np
import pandas as pd

# =============================================================
# SQL / DATABASE CONNECTIVITY
# =============================================================
import pyodbc

# =============================================================
# MACHINE LEARNING / VECTORIZATION
# =============================================================
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ========================================================================
# NLP / TRANSFORMERS / HTML PARSING
# ========================================================================
import spacy
from spacy.cli import download as spacy_download
from bs4 import BeautifulSoup
from bs4 import MarkupResemblesLocatorWarning
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer

# ========================================================================
# OPTIONAL VISUALIZATION + REQUESTS (for future use)
# ========================================================================
import matplotlib.pyplot as plt
import requests

# ========================================================================
# OPTIONAL LOCAL LIBRARIES (if applicable)
# ========================================================================
from pandas.io.json._table_schema import build_table_schema
from lmdatalink.databases import MSSQLConnection  # Only if needed for other connections

# ========================================================================
# WARNINGS HANDLING
# ========================================================================
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="DataFrame.applymap has been deprecated.*")
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*toDlpack.*")

# ========================================================================
# Elastic Search and Transformers
# ========================================================================
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, BulkIndexError, scan
from transformers import AutoTokenizer
import tiktoken

# ========================================================================
# GPT 
# ========================================================================
from openai import OpenAI
from httpx import Client
import openai


@dataclass
class ScoreCardConfig:
    # ==========================================================================
    # GLOBAL CONFIG / EXECUTION SETTINGS
    # ==========================================================================
    spacy_model:            str = "en_core_web_trf"                             # The SpaCy Model to use
    json_path:              str = "../scorecard_state/state.json"                  # Path for saving state snapshots
    model_matrix_json:      str = "../prompts/model_matrix.json"                           # Path to model config JSON
    vectorizer_strategy:    Literal["count", "tfidf", "binary"] = "count"       # Text vectorizer type
    
    # ==========================================================================
    # SQL CONNECTION SETTINGS
    # ==========================================================================
    sql_driver_path:        str = "/home/jovyan/local/msodbcsql18/opt/microsoft/msodbcsql18/lib64/libmsodbcsql-18.3.so.2.1"  # ODBC driver path
    sql_server:             str = "EDWSQL40.us.lmco.com\\SSAPPPRD"              # SQL server name
    sql_database:           str = "SubcontractScorecard"                        # Target database
    sql_uid:                str = "ACE_TEAM"                                    # SQL user
    sql_pwd:                str = "TGPXKUD_23Pn"                                # SQL password
    sql_query_file:         str = "../prompts/sql_query.txt"                               # File path for SQL query
    
    # ==========================================================================
    # EXECUTION SWITCHES (enable/disable pipeline stages)
    # ==========================================================================
    require_gpu:        bool = True                                             # Reserve GPU use if available or required later
    sql_download:       bool = True                                             # Download from SQL; if False, loads from Elasticsearch
    enable_nlp:         bool = True                                             # Run spaCy NLP enrichment and build SID windows
    build_models:       bool = True                                              # Train models on enriched and prepared features
    run_predictions:    bool = True                                              # Apply best model to full dataset for inference
    build_rag:          bool = True                                            # Save time if I don't need it

    # ==========================================================================
    # ELASTICSEARCH CONFIGURATION
    # ==========================================================================
    es_host:                str = "http://localhost:9200"                       # Local Elasticsearch endpoint
    elastic_index:          str = "scorecard"                                   # Index name for enriched documents
    tokenizer_name:         str = "cl100k_base"                                 # GPT tokenizer to use for lexical fallback
    
    # ==========================================================================
    # EMBEDDING MODEL CONFIGURATION
    # ==========================================================================
    embedding_model_name:   str = "BAAI/bge-large-en-v1.5"                      # Embedding model name
    embedding_vector_dim:   int = 1024                                          # Output vector dimensionality
    rag_index:              str = "scorecard_rag_notes"                         # You know for search
    batch_size:             int = 128                                           # The chocolate coating makes it go down easier

    # ==========================================================================
    # Modeling
    # ==========================================================================
    training_length:        int = 5                                             # Number of notes needed to train model

    # ==========================================================================
    # GPT CONFIGURATION
    # ==========================================================================
    gpt_prompt_location:    str = "../prompts/GPT_Prompt.txt"                              # System prompt file for GPT
    gpt_base_url:           str = "https://api.ai.us.lmco.com/v1/"              # LMI GPT API base URL

    # ==========================================================================
    # REPORTING + LOGGING
    # ==========================================================================
    log_index_name:         str = "scorecard_log"                               # ES index for pipeline logs
    log_file_path:          str = "logs/scorecard_log.txt"                      # Local file for debug logs

    def __post_init__(self):
        """D - D - D - Defence """
        assert self.training_length >= 2, (
            f"Invalid training_length: {self.training_length}. "
            "Must be at least 2 to include 1+ historical notes."
        )

@dataclass
class ScoreCardState:
    # ==========================================================================
    # The Locations from above to load out all of the good stuff
    # ==========================================================================
    config:                 ScoreCardConfig                                     # All the config locations
    # ==========================================================================
    # RAW + ENRICHED DATAFRAMES (generated during pipeline stages)
    # ==========================================================================
    raw_df:                 Optional[pd.DataFrame] = None                       # Original raw note dataset
    details_df:             Optional[pd.DataFrame] = None                       # SQL-joined metadata
    enriched_df:            Optional[pd.DataFrame] = None                       # NLP-enriched note content
    sid_df:                 Optional[pd.DataFrame] = None                       # SID note history windows
    predicted_sids:         Optional[pd.DataFrame] = None                       # SID + label predictions
    complete_dataset:       Optional[pd.DataFrame] = None                       # Modeling dataframe
    predictions_df:         Optional[pd.DataFrame] = None                       # Note-level predictions
    final_df:               Optional[pd.DataFrame] = None                       # Final DataFrame

    # ==========================================================================
    # TRAINED MODELS + RESULT TRACKING
    # ==========================================================================
    models:                 dict[str, dict[str, Any]] = field(default_factory=dict)  # Trained models by key
    best_model:             Optional[dict[str, Any]] = None                      # Best performing model config
    best_model_key:         Optional[str] = None                                 # Best model lookup key
    model_grid_df:          Optional[pd.DataFrame] = None                        # Grid of model hyperparams
    prepared_datasets:      Optional[dict[str, dict[str, Any]]] = None           # Sampled + full matrices

    # ==========================================================================
    # LOADED MODEL MATRIX CONFIGS (feature / sampling / vectorization)
    # ==========================================================================
    feature_matrix:         Optional[dict[str, dict[str, list[str]]]] = None    # Feature set definitions
    sampling_matrix:        Optional[dict[str, dict[str, Any]]] = None          # Sampling strategies
    vectorization_matrix:   Optional[dict[str, dict[str, str]]] = None          # Vectorizer + encoder strategy
    model_weights:          Optional[list[dict[int, float]]] = None             # Class weighting strategies
    text_field_config:      Optional[dict[str, list[str]]] = None               # Text fields per feature set
    numeric_fields:         Optional[list[str]] = None                          # Numeric fields for input
    model_configs:          Optional[list[dict[int, float]]] = None             # Class weight dictionaries

    # ==========================================================================
    # MODEL PIPELINE COMPONENTS
    # ==========================================================================
    vectorizer:             Optional[Any] = None                                # Fitted vectorizer
    X_matrix:               Optional[Any] = None                                # Final input matrix (X)
    y_vector:               Optional[Any] = None                                # Final target vector (y)
    tokenizer:              Optional[Any] = None                                # GPT tokenizer
    nlp:                    Optional[spacy.language.Language] = None            # Loaded spaCy NLP pipeline

    # ==========================================================================
    # EMBEDDING + GPT OBJECTS
    # ==========================================================================
    embedding_model:        Optional[SentenceTransformer] = None                # SentenceTransformer model
    embedding_dim:          Optional[int] = None                                # Embedding dimensionality (e.g. 384 or 768)

    def __post_init__(self) -> None:
        
        self._load_model_matrices()
        self._initialize_spacy_model()
        self._load_sql_query()
        self._load_prompt_template()

    def _load_model_matrices(self) -> None:
        """
        Loads model feature, sampling, and vectorization matrices from a single JSON config file.
        """
        with open(self.config.model_matrix_json, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.feature_matrix = config.get("feature_matrix", {})
        self.sampling_matrix = config.get("sampling_matrix", {})
        self.vectorization_matrix = config.get("vectorization_matrix", {})
        self.model_weights = config.get("model_weights", None)
        self.vectorizer_params = config.get("vectorizer_params", {})

        # Need to convert this to a Tupple...because...shut up..JSONs can't handle tupples...what?
        self.vectorizer_params["ngram_range"] = tuple(self.vectorizer_params["ngram_range"])
    
    def _initialize_spacy_model(self) -> None:

        try:
            self.nlp = spacy.load(self.config.spacy_model)
        except OSError:
            print(f"Model '{self.config.spacy_model}' not found. Downloading it now...")
            spacy_download(self.config.spacy_model)
    
    def _load_sql_query(self) -> None:
        """Loads the SQL query from sql_query.txt. Fails loudly if not found."""

        with open(self.config.sql_query_file, encoding="utf-8") as f:
            self.sql_query = f.read().strip()

    def _load_prompt_template(self) -> str:
        """
        Loads GPT system prompt from file if not already loaded.
        Stores result in `self.gpt_prompt`.
        """

        with open(self.config.gpt_prompt_location, "r", encoding="utf-8") as f:
            self.gpt_prompt = f.read()
    
        return self.gpt_prompt

class ConnectionManager:
    def __init__(self, config: ScoreCardConfig, state: ScoreCardState) -> None:
        self.config = config
        self.state = state
        self.log_file_path = config.log_file_path
        self.log_index_name = config.log_index_name

        # Core client handles
        self.sql_conn = None
        self.es_client = None
        self.gpt_client = None

        # === Connect in proper order ===
        self._connect_elasticsearch()  # Needed before using self.report
        self._connect_sql()
        self._connect_gpt()
        self._load_embedding_model()

    def _connect_sql(self) -> None:
        conn_str = (
            f"DRIVER={{{self.config.sql_driver_path}}};"
            f"SERVER={self.config.sql_server};"
            f"DATABASE={self.config.sql_database};"
            f"UID={self.config.sql_uid};"
            f"PWD={self.config.sql_pwd}"
        )
        self.sql_conn = pyodbc.connect(conn_str)
        assert self.sql_conn is not None, "SQL connection returned None"
        self.state.sql_connection = self.sql_conn
        self.report("CONN", "SQL connection established.")

    def _connect_elasticsearch(self) -> None:
        self.es_client = Elasticsearch(self.config.es_host)
        assert self.es_client.ping(), "Elasticsearch ping failed"
        self.state.es_conn = self.es_client
        self.report("CONN", "Elasticsearch connection established.")

    def _connect_gpt(self) -> None:
        self.gpt_client = OpenAI(
            base_url=self.config.gpt_base_url,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        assert self.gpt_client is not None, "GPT client initialization failed"
        self.state.gpt_client = self.gpt_client
        self.report("CONN", "GPT client initialized.")

    def _load_embedding_model(self) -> None:
        """
        Loads the sentence embedding model onto GPU and stores it in state.
        Embedding dimension is also stored for reference (e.g., zero vector fallback).
        """
        if not torch.cuda.is_available():
            raise RuntimeError("[EMBD] GPU not available — cannot load embedding model")

        self.report("EMBD", f"Loading embedding model '{self.config.embedding_model_name}' to GPU")
        model = SentenceTransformer(self.config.embedding_model_name)
        model = model.to("cuda")

        self.state.embedding_model = model
        self.state.embedding_dim = model.get_sentence_embedding_dimension()
        self.report("EMBD", f"Embedding model loaded with dim {self.state.embedding_dim}")

    def report(self, tag: str, message: str, meta: dict = None) -> None:
        timestamp = datetime.utcnow().isoformat()
        full_message = f"[{tag.upper()}] \t{message}"
        print(full_message)

        # Log to file
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        with open(self.log_file_path, "a") as f:
            f.write(f"{timestamp} {full_message}\n")

        # Log to Elasticsearch
        if self.es_client:
            self.es_client.index(index=self.log_index_name, document={
                "timestamp": timestamp,
                "tag": tag.upper(),
                "message": message,
                "meta": meta or {}
            })

    @staticmethod
    def is_scalar(val: Any) -> bool:
        return isinstance(val, (
            str, int, float, bool, type(None),
            pd.Timestamp, datetime, np.integer, np.floating
        ))

    @staticmethod
    def is_valid_list(val: Any) -> bool:
        if isinstance(val, (list, tuple, np.ndarray)):
            return all(ConnectionManager.is_scalar(v) or pd.isna(v) for v in val)
        return False

    @staticmethod
    def clean(val: Any) -> Any:
        """
        Converts a Python object into a JSON-safe primitive for Elasticsearch.

        Raises:
            TypeError: If the value cannot be safely serialized.
        """
        if isinstance(val, (pd.Timestamp, datetime)):
            return val.isoformat()

        if isinstance(val, (np.integer, np.floating)):
            return val.item()

        if ConnectionManager.is_scalar(val) and pd.isna(val):
            return None

        if isinstance(val, np.ndarray):
            return [ConnectionManager.clean(v) for v in val.tolist()]

        if isinstance(val, (list, tuple)):
            return [ConnectionManager.clean(v) for v in val]

        if ConnectionManager.is_scalar(val):
            return val

        raise TypeError(f"Unsupported type for serialization: {type(val)} | Value: {repr(val)}")

    def serialize_row_for_es(self, row: pd.Series, drop_keys: list[str] = None) -> dict:
        drop_keys = set(drop_keys or [])
        drop_keys.add("_id")  # Always drop ES-reserved field

        result = {}
        for k, v in row.items():
            if k in drop_keys:
                continue
            if self.is_scalar(v) or self.is_valid_list(v):
                result[k] = self.clean(v)
            # else: unsupported types will raise inside clean

        return result

    def upload_dataframe_to_es(self, df: pd.DataFrame, index_name: str, id_col: Optional[str] = None) -> None:
        """
        Uploads a DataFrame to Elasticsearch using bulk indexing. Each row is converted
        into a document, and its ES document ID is either taken from a specified column
        or from the DataFrame index.

        This function will delete and recreate the index before uploading, so any existing
        data in the target index will be lost. It also includes a zero embedding vector
        of appropriate dimension if one is not already present.

        Parameters:
            df (pd.DataFrame): The DataFrame to upload.
            index_name (str): The name of the Elasticsearch index to write to.
            id_col (str, optional): The column in the DataFrame to use as the ES _id field.
                If None, the DataFrame index is used instead.

        Raises:
            ValueError: If `id_col` is specified but not present in the DataFrame.
            RuntimeError: If any documents fail to index.
        """

        # Ensure an ID column is available
        if id_col is not None:
            assert id_col in df.columns, f"id_col '{id_col}' not found in DataFrame columns."
            df["_id"] = df[id_col].astype(str)
        else:
            df["_id"] = df.index.astype(str)

        # Add placeholder embedding if not already present
        if "embedding" in df.columns:
            self.embeded_final_dataframe()
            return 

        # Delete + recreate index
        if self.es_client.indices.exists(index=index_name):
            self.es_client.indices.delete(index=index_name)
        self.es_client.indices.create(index=index_name)

        # Build bulk actions
        actions = []
        for _, row in df.iterrows():
            actions.append({
                "_index": index_name,
                "_id": row["_id"],
                "_source": self.serialize_row_for_es(row)
            })

        # Upload to Elasticsearch
        success, errors = bulk(
            self.es_client,
            actions,
            stats_only=False,
            raise_on_error=False,
            raise_on_exception=False
        )

        if errors:
            self.report("ES", f"{len(errors)} document(s) failed to index.")
            for error in errors[:5]:  # Show only the first few for brevity
                print("-" * 60)
                print(json.dumps(error, indent=2))
            raise RuntimeError(f"Bulk upload to '{index_name}' failed for {len(errors)} document(s).")

        self.report("ES", f"Indexed {success} documents to '{index_name}' with 0 errors.")

    def load_from_es(self, index_name: str, id_col: str = "sid") -> pd.DataFrame:
        self.report("ES", f"Loading data from index '{index_name}'...")

        query = {
            "query": {"match_all": {}},
            "_source": True
        }

        scroll = scan(
            client=self.es_client,
            index=index_name,
            query=query,
            preserve_order=True,
            scroll="2m"
        )

        rows = []
        for doc in scroll:
            row = doc["_source"]
            row[id_col] = doc["_id"]
            rows.append(row)

        df = pd.DataFrame(rows)
        self.report("ES", f"Loaded {len(df)} rows from '{index_name}'")
        return df

    def embeded_final_dataframe(self):
        pass

class ScoreCardTextPrep:
    """
    Handles all text-based preparation steps for ScoreCard modeling.

    Responsibilities:
    - HTML cleaning and SID key generation
    - SpaCy-based NLP enrichment (verbs, adjectives, noun chunks)
    - Main word extraction and lowercase normalization
    - Target label creation and next color mapping
    - SID-based sliding history window construction for model training

    All operations are applied to a shared ScoreCardState object and stored
    back into its `enriched_df` and `sid_df` attributes.
    """
    def __new__(cls, *args, **kwargs):
        raise TypeError("ScoreCardTextPrep is a static utility class and cannot be instantiated.")

    @staticmethod
    def enrich(state: ScoreCardState) -> None:
        """
        Performs HTML-to-text cleaning, extracts linguistic features using SpaCy,
        generates main_words, and maps label fields on the `details_df`.
        Results are stored in `state.enriched_df`.
        """
        df = state.details_df.copy()

        df["sid_key"] = (
            df["SID"].astype(str).str.zfill(6) + "." +
            df["Note_Year"].astype(str) + "." +
            df["Note_Month"].astype(str) + "." +
            df["Scorecard_Detail_Note_SID"].astype(str).str.zfill(6)
        )

        df = df.sort_values(by="sid_key", ascending=True).reset_index(drop=True)
        df["pre_scrub_text"] = df["Scorecard_Note"].apply(ScoreCardTextPrep._html_to_text)
        df["Scorecard_Note"] = df["pre_scrub_text"].copy()

        docs = list(state.nlp.pipe(df["pre_scrub_text"].tolist(), disable=["ner"]))
        df["verbs"] =       [" ".join(t.lemma_ for t in doc if t.pos_ == "VERB") for doc in docs]
        df["adjectives"] =  [" ".join(t.lemma_ for t in doc if t.pos_ == "ADJ")  for doc in docs]
        df["noun_chunks"] = [" ".join(chunk.text for chunk in doc.noun_chunks) for doc in docs]

        df["main_words"] = (
            df["verbs"] + " " + df["adjectives"] + " " + df["noun_chunks"]
        ).str.lower().replace(r"\s+", " ", regex=True)

        label_map = defaultdict(lambda: -1, {"G": 0, "Y": 1, "R": 2})
        df = df.dropna(subset=["Overall"])
        df["target"] = df["Overall"].map(label_map)

        df["next_color_code"] = (
            df.groupby("SID")["Overall"]
              .shift(-1)
              .map(label_map)
              .fillna(-1)
              .astype(int)
        )

        grouped = df.groupby("SID", group_keys=False, sort=False)
        df["note_history"] = grouped["Scorecard_Detail_Note_SID"].apply(
            lambda col: pd.Series(
                [""] + [";".join(col.astype(str).iloc[:i]) for i in range(1, len(col))],
                index=col.index
            )
        )

        state.enriched_df = df

    @staticmethod
    def build_sid_history(state: ScoreCardState) -> None:
        """
        Constructs per-SID sliding windows of note history for training and inference.

        Each sliding window becomes one row in `sid_df`, capturing:
            - Up to 4 prior notes as model input (main_words, sid_key, color)
            - The final (5th) note as the prediction target (text, color, metadata)
            - Flag indicating if the SID had > 4 notes (more_than_4)
            - Total number of notes in the SID (total_notes)
            - Whether all 4 prior notes were green (all_green)
            - Concatenated `complete_main_words` string for model input

        Notes:
        - The function respects true chronological order via `sid_key`.
        - Windows are of size 5 when possible: 4 inputs + 1 target.
        - For SID histories with fewer than 5 notes, partial windows are still used for prediction,
        but not for training (i.e., target = -1).
        - The final note in each SID is always included for inference purposes, even if
        there is no known future label.
        - Each row's `last_sid` refers to the most recent input note, not the prediction target.
        - The target label is only included when the window is exactly 5 notes long.

        Sliding Windows, Inputs, and Prediction Rows:

            Window               Input Notes         Target Note   last_sid   Trainable   Target Available?
            ------------------- ------------------- ------------- ---------- ----------- -------------------
            [1, 2]              [1]                 2             1         [no]        [yes]
            [1, 2, 3]           [1, 2]              3             2         [no]        [yes]
            [1, 2, 3, 4]        [1, 2, 3]           4             3         [no]        [yes]
            [1, 2, 3, 4, 5]     [1, 2, 3, 4]        5             4         [yes]       [yes]
            [2, 3, 4, 5, 6]     [2, 3, 4, 5]        6             5         [yes]       [yes]
            [3, 4, 5, 6, 7]     [3, 4, 5, 6]        7             6         [yes]       [yes]
            [4, 5, 6, 7, 8]     [4, 5, 6, 7]        8             7         [yes]       [yes]
            [5, 6, 7, 8]        [5, 6, 7, 8]        ???           8         [no]        [no] (future note unknown)
            [6, 7, 8]           [6, 7, 8]           ???           8         [no]        [no] 9 to predict <---- I will add this soon

        All rows in `sid_df` are useful for inference (RAG or model input),
        but only those with 5-note windows are valid training examples.

        Steps:
        1. Group `enriched_df` by SID, preserving chronological order using `sid_key`.

        2. For each SID:
            a. Generate sliding windows of notes:
                - Include all windows of size 2 to 5 (inclusive) when a future note exists.
                - Include one final prediction-only window of size 4 if it ends the SID with no future.

            b. For each window:
                i.   Record `sid`, `more_than_4` (1 if total notes ≥ 5), and `total_notes`.
                ii.  Identify input notes: all but the final note in the window.
                iii. Identify the target note: final note in the window.
                iv.  Set `trainable = True` if the window has exactly 5 notes and a future label.
                v.   Set `target = -1` if no future note exists.
                vi.  Call `_populate_input_fields()` to store main_words and colors from prior notes.
                vii. Call `_populate_target_fields()` to store target color and last_sid.
                viii.Copy supplier and contract metadata from the target note.
                ix.  Derive additional fields:
                    - `color_set`: joined colors from input notes
                    - `all_green`: 1 if all input notes are "G"
                    - `complete_main_words`: combined main_words + metadata string

        3. Compile all window entries into `state.sid_df` as a new DataFrame. This function killed me for a while
        So I really needed to document it...lol...¯\(ツ)/¯
        """

        # Step 1: Group `enriched_df` by SID, preserving order using `sid_key`
        df = state.enriched_df.copy()

        sid_to_notes = {}
        for _, row in df.iterrows():
            sid_to_notes.setdefault(row["SID"], []).append(row)

        # Step 2: For each SID
        rows = {}
        key = 0

        for sid, notes in sid_to_notes.items():
            total = len(notes)

            # Step 2a: Generate sliding windows for this SID
            windows = ScoreCardTextPrep._generate_note_windows(notes, max_window_size=5)
            more_than_4 = int(total >= 5)

            # Step 2b: For each window
            for window in windows:
                # Step 2b.i: Initialize entry with base metadata
                entry = {
                    "sid": sid,
                    "more_than_4": more_than_4,
                    "total_notes": total
                }

                # Step 2b.ii: Identify input notes (all but the final one)
                input_notes = window["Input_notes"]

                # Step 2b.iii: Identify target note (may be None for final prediction-only window)
                target_note = window["target_note"]

                # Step 2b.iv: Identify last note in inputs (used for fallback join key)
                last_note = window["last_note"]

                # Step 2b.v: Determine if this window is trainable (exactly 5 notes and has target)
                entry["trainable"] = trainable = window["trainable"]

                # Step 2b.vi: Populate input fields: sid_key_i, main_words_i, Color_Code_i
                color_set, main_words_list = ScoreCardTextPrep._populate_input_fields(entry, input_notes)

                # Step 2b.vii: Populate target fields (last_sid, target, Color_Code_4, etc.)
                fallback_sid = last_note["sid_key"]
                if target_note is not None:
                    populated_target = ScoreCardTextPrep._populate_target_fields(
                        entry,
                        note=target_note,
                        sid_key_fallback=fallback_sid,
                        trainable=trainable
                    )
                else:
                    # Fallback values when no target note (final note inference only)
                    entry["last_sid"] = fallback_sid
                    entry["target"] = -1
                    entry["Color_Code_4"] = ""
                    entry["last_note_text"] = last_note["Scorecard_Note"]
                    entry["last_note_main_words"] = last_note["main_words"]
                    populated_target = last_note

                # Step 2b.viii: Copy supplier and contract metadata from target/fallback
                meta = {
                    "LM_Vendor_ID":                populated_target["LM_Vendor_ID"],
                    "Performance_Management_Type": populated_target["Performance_Management_Type"],
                    "PO_Complexity_Level":         populated_target["PO_Complexity_Level"],
                    "PO_Contract_Dollars_Mil":     populated_target["PO_Contract_Dollars_Mil"],
                    "Supplier_Name":               populated_target["Supplier_Name"],
                    "LOB_Name":                    populated_target["LOB_Name"]
                }
                entry.update(meta)

                # Step 2b.ix.1: Derive `color_set` and `all_green` flag
                entry["color_set"] = color_set
                entry["all_green"] = int(len(color_set) == 4 and color_set == "G" * 4)

                # Step 2b.ix.2: Construct `complete_main_words` input string
                vendor     = meta["LM_Vendor_ID"]
                lob        = f"lob_{meta['LOB_Name'].replace(' ', '_')}"
                name       = f"name_{meta['Supplier_Name'].replace(' ', '_')}"
                complexity = f"complexity_{meta['PO_Complexity_Level']}"
                management = f"management_{meta['Performance_Management_Type']}"
                complete_main = " ".join(main_words_list).strip()
                entry["complete_main_words"] = f"{complete_main} {vendor} {lob} {name} {complexity} {management}".strip()

                # Step 2b.x: Store window entry
                rows[key] = entry
                key += 1

        # Step 3: Compile all windows into a DataFrame
        state.sid_df = pd.DataFrame.from_dict(rows, orient="index")

    @staticmethod
    def _generate_note_windows(notes: list[pd.Series], max_window_size: int = 5) -> list[dict]:
        """
        Returns structured note windows from a list of note rows (pd.Series), for a given SID.

        Each window includes:
            - Input_notes: all but the last note in the window
            - target_note: final note (if known)
            - last_note: the most recent input note
            - trainable: True if this is a 5-note window with a known target

        Args:
            notes (list of pd.Series): Chronologically ordered list of notes for a SID
            max_window_size (int): Max window size (default 5)

        Returns:
            list of dicts: Each dict contains a full window definition
        """
        results = []
        n = len(notes)

        for i in range(1, n + 1):
            if i + 1 <= max_window_size:
                window = notes[:i+1]
            else:
                window = notes[i+1 - max_window_size:i+1]

            input_notes = window[:-1]
            last_note = input_notes[-1] if input_notes else None

            if i < n:
                target_note = window[-1]
            else:
                # Final partial window — used only for inference
                input_notes = notes[-max_window_size + 1:]
                target_note = None
                last_note = input_notes[-1]

            results.append({
                "Input_notes": input_notes,
                "target_note": target_note,
                "last_note": last_note,
                "trainable": not (len(input_notes) < max_window_size - 1 or target_note is None)
            })

        return results

    @staticmethod
    def _populate_input_fields(entry: dict, notes: list) -> tuple[str, list[str]]:
        color_seq = ""
        main_words_seq = []
        for i, note in enumerate(notes):
            entry[f"sid_key_{i}"] = note["sid_key"]
            entry[f"main_words_{i}"] = note["main_words"]
            color = note["Overall"]
            entry[f"Color_Code_{i}"] = color
            color_seq += color
            main_words_seq.append(note["main_words"])
        return color_seq, main_words_seq

    @staticmethod
    def _populate_target_fields(
        entry: dict,
        note: dict,
        sid_key_fallback: str,
        trainable: bool = True
    ) -> dict:
        """
        Populates target-related fields for the final note in a sliding window.

        Parameters:
            entry (dict): The current row dictionary being assembled.
            note (dict): The target note (last note in the window).
            sid_key_fallback (str): The sid_key of the last input note.
            trainable (bool): Whether this row has a known future label.

        Returns:
            dict: The same `note` dict (for metadata extraction).
        """
        entry["last_sid"] = sid_key_fallback
        entry["target"] = note.get("target", -1)  # Target should reflect label availability, not trainability
        entry["Color_Code_4"] = note.get("Overall", "") if trainable else ""
        entry["last_note_text"] = note.get("Scorecard_Note", "")
        entry["last_note_main_words"] = note.get("main_words", "")

        # Optional debug warning if you ever want to catch odd edge cases:
        # if not trainable and note.get("target", -1) != -1:
        #     warnings.warn(f"Non-trainable note has a valid label: {note.get('sid_key', '?')}")

        return note

    @staticmethod
    def _html_to_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(['p', 'li']):
            text = tag.get_text(strip=True)
            if text and not re.search(r'[.!?]\s*$', text) and len(text.split()) > 2:
                tag.append('.')
        for br in soup.find_all('br'):
            br.replace_with(' ')
        text = soup.get_text(separator=' ')
        text = re.sub(r'\s*\.\s*', '. ', text)
        text = re.sub(r'\.\s*\.+', '. ', text)
        text = re.sub(r'([:;])\s*\.', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'&(?:[a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', text)
        text = re.sub(r'[\n\r\t]+', ' ', text)
        return text.strip()

class ScoreCardModeling:
    def __init__(self, config: ScoreCardConfig, state: ScoreCardState, conn: ConnectionManager) -> None:
        self.config = config
        self.state = state
        self.conn = conn
        self.max_workers = os.cpu_count() // 4 or 1

    def build_model_grid(self) -> None:
        """
        Generates the model configuration grid from the Cartesian product of feature, sampling, and vectorizer options.

        Builds:
            - self.state.model_grid_df: A DataFrame containing all valid combinations of:
                - feature_set
                - sampling_strategy
                - vectorizer_strategy

        Rules:
            - If a feature set has no text fields, it only allows vectorizers with 'count' type.
        """
        grid = []
        for feat_name, feat_cfg in self.state.feature_matrix.items():
            has_text = bool(feat_cfg.get("text_fields"))
            for sample_name in self.state.sampling_matrix:
                for vec_name, vec_cfg in self.state.vectorization_matrix.items():
                    if not has_text and vec_cfg["text_vectorizer"] != "count":
                        continue
                    grid.append({
                        "feature_set": feat_name,
                        "sampling_strategy": sample_name,
                        "vectorizer_strategy": vec_name
                    })
        self.state.model_grid_df = pd.DataFrame(grid)

    def prepare_dataset(self, sid_df: pd.DataFrame, config_row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Constructs a training dataset (X, y) from a specific model configuration.

        Args:
            sid_df (pd.DataFrame): Input dataframe with enriched scorecard notes.
            config_row (dict[str, Any]): A row from the model grid containing:
                - 'feature_set'
                - 'sampling_strategy'
                - 'vectorizer_strategy'

        Returns:
            X (np.ndarray): Final feature matrix.
            y (np.ndarray): Target label array.
            metadata (dict): Fitted transformers (vectorizer, scaler, encoders) and config references.
        """

        feature_key = config_row["feature_set"]
        sampling_key = config_row["sampling_strategy"]
        vectorizer_key = config_row["vectorizer_strategy"]

        features = self.state.feature_matrix[feature_key]
        sampling = self.state.sampling_matrix[sampling_key]
        vectorization = self.state.vectorization_matrix[vectorizer_key]

        text_fields = features.get("text_fields", [])
        numeric_fields = features.get("numeric_fields", [])
        categorical_fields = features.get("categorical_fields", [])

        # Track total before filtering
        total_notes = len(sid_df)

        # Filter to rows with sufficient history and marked as trainable
        sid_df = sid_df[
            (sid_df["total_notes"] >= self.config.training_length) &
            (sid_df["trainable"] == True) &
            (sid_df["target"].isin([0, 1, 2]))
        ].copy()

        # Calculate % culled
        kept_notes = len(sid_df)
        if total_notes > 0:
            percent_culled = 100 * (total_notes - kept_notes) / total_notes
        else:
            percent_culled = 0.0

        self.conn.report("DATA", (
            f"Filtered to {kept_notes} trainable rows with total_notes >= {self.config.training_length} "
            f"and valid target (0/1/2) {percent_culled:.1f}% culled"
        ))

        if sampling["method"] == "downsample":
            majority = sid_df[sid_df["all_green"] == 1].sample(frac=1, random_state=42)
            minority = sid_df[sid_df["all_green"] == 0]
            keep = min(len(minority) * sampling["majority_multiplier"], len(majority))
            df = pd.concat([majority.iloc[:keep], minority]).sample(frac=1, random_state=42)
        else:
            df = sid_df.copy()

        y = df["target"].values

        parts = []
        metadata = {
            "feature_set": feature_key,
            "sampling": sampling_key,
            "vectorizer": vectorizer_key,
            "text_vectorizer": None,
            "numeric_scaler": None,
            "cat_encoder": None,
            "text_fields": text_fields,
            "numeric_fields": numeric_fields,
            "categorical_fields": categorical_fields
        }

        if text_fields:
            vectorizer_type = vectorization["text_vectorizer"]
            vec_cls = CountVectorizer if vectorizer_type == "count" else TfidfVectorizer
            vectorizer = vec_cls(**vectorization.get("vectorizer_params", {}))
            corpus = df[text_fields[0]].fillna("").astype(str).tolist()
            X_text = vectorizer.fit_transform(corpus).toarray()
            parts.append(X_text)
            metadata["text_vectorizer"] = vectorizer

        if numeric_fields:
            scaler = StandardScaler()
            X_numeric = scaler.fit_transform(df[numeric_fields].fillna(0))
            parts.append(X_numeric)
            metadata["numeric_scaler"] = scaler

        if categorical_fields:
            if vectorization.get("cat_encoder") == "label":
                encoded_cols = []
                encoders = {}
                for col in categorical_fields:
                    le = LabelEncoder()
                    encoded = le.fit_transform(df[col].astype(str))
                    encoded_cols.append(encoded.reshape(-1, 1))
                    encoders[col] = le
                X_cat = np.hstack(encoded_cols)
                metadata["cat_encoder"] = encoders
            else:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                X_cat = ohe.fit_transform(df[categorical_fields].astype(str))
                metadata["cat_encoder"] = ohe
            parts.append(X_cat)

        X = np.hstack(parts) if parts else np.empty((len(df), 0))
        return X, y, metadata

    def prepare_prediction_matrix(self, sid_df: pd.DataFrame, metadata: dict[str, Any]) -> np.ndarray:
        """
        Generates a feature matrix (X) for prediction using a pre-fit model's metadata.

        Args:
            sid_df (pd.DataFrame): The input dataframe for prediction (same schema as training).
            metadata (dict): Metadata from a trained model containing transformers:
                - text_vectorizer
                - numeric_scaler
                - cat_encoder

        Returns:
            X (np.ndarray): Feature matrix ready for model.predict or model.predict_proba.
        """

        parts = []

        if metadata["text_vectorizer"] and metadata["text_fields"]:
            vec = metadata["text_vectorizer"]
            corpus = sid_df[metadata["text_fields"][0]].fillna("").astype(str).tolist()
            parts.append(vec.transform(corpus).toarray())

        if metadata["numeric_scaler"] and metadata["numeric_fields"]:
            scaler = metadata["numeric_scaler"]
            X_numeric = scaler.transform(sid_df[metadata["numeric_fields"]].fillna(0))
            parts.append(X_numeric)

        if metadata["cat_encoder"] and metadata["categorical_fields"]:
            encoder = metadata["cat_encoder"]
            if isinstance(encoder, dict):
                encoded_cols = []
                for col, le in encoder.items():
                    encoded = le.transform(sid_df[col].astype(str))
                    encoded_cols.append(encoded.reshape(-1, 1))
                X_cat = np.hstack(encoded_cols)
            else:
                X_cat = encoder.transform(sid_df[metadata["categorical_fields"]].astype(str))
            parts.append(X_cat)

        return np.hstack(parts) if parts else np.empty((len(sid_df), 0))

    def find_best_model(self) -> None:
        """
        Trains and evaluates logistic regression models for each combination in the model grid and model_weights.

        Process:
            - Calls prepare_dataset(...) for each model config.
            - Trains a logistic regression with the specified class weights.
            - Scores each model by total false negatives (y == 1 or 2 misclassified as 0).
            - Tie-breaks by highest test set accuracy if false negatives are equal.
            - Selects the best model and saves it to:
                - self.state.best_model
                - self.state.best_model_key

        Also populates:
            - self.state.models: All models and their training stats/errors.
        """

        results = {}

        # Normalize weights to ensure keys are integers
        int_weights = [{int(k): v for k, v in weight.items()} for weight in self.state.model_weights]

        # Generate all combinations of (config_row, weight)
        tasks = [
            (row, weight)
            for _, row in self.state.model_grid_df.iterrows()
            for weight in int_weights
        ]

        self.conn.report("MODL", f"Training {len(tasks)} models serially with LogisticRegression(n_jobs=-1)")

        # Serial model training
        for row, weight in tasks:
            try:
                dataset_key = f"{row['feature_set']} | {row['sampling_strategy']} | {row['vectorizer_strategy']}"
                self.conn.report(
                    "MODL",
                    f"Training model: {dataset_key} | {weight} | FN={result['total_false_negatives']} | Acc={result['accuracy']:.4f}"
                )

                X, y, meta = self.prepare_dataset(self.state.sid_df, row)
                dataset = {"X": X, "y": y, "metadata": meta}

                model_id, result = self.train_one_model(dataset_key, dataset, weight, n_jobs=-1)
                results[model_id] = result

                if result.get("model"):
                    self.conn.report("MODL", f"Completed: {model_id} | FN={result['total_false_negatives']} | Acc={result['accuracy']:.4f}")
                else:
                    self.conn.report("MODL", f"Error in: {model_id} | {result.get('error')}")

            except Exception as e:
                model_id = f"{row['feature_set']} | {row['sampling_strategy']} | {row['vectorizer_strategy']} | {weight}"
                self.conn.report("MODL", f"Training failed for model: {model_id}. Error: {str(e)}")
                results[model_id] = {
                    "model": None,
                    "error": str(e),
                    "total_false_negatives": None
                }

        # Store all results
        self.state.models = results

        # Filter valid models
        valid_models = {
            k: v for k, v in results.items()
            if v.get("model") and v.get("total_false_negatives") is not None
        }

        # Select best model
        if valid_models:
            best_key = min(
                valid_models,
                key=lambda k: (
                    valid_models[k]["total_false_negatives"],
                    -valid_models[k]["accuracy"]
                )
            )
            self.state.best_model = valid_models[best_key]
            self.state.best_model_key = best_key
            self.conn.report("MODL", f"Selected best model: {best_key} | FN={valid_models[best_key]['total_false_negatives']}")
        else:
            self.state.best_model = None
            self.state.best_model_key = None

        assert self.state.best_model_key, "Error! No model found"

    def train_one_model(
        self,
        dataset_key: str,
        dataset: dict[str, Any],
        class_weights: dict[int, float],
        n_jobs = -1) -> tuple[str, dict[str, Any]]:
        """
        Trains a logistic regression model on a single dataset/config.
        Returns model ID and result dictionary.
        Raises immediately on error — no silent failures.
        """
        X = dataset["X"]
        y = dataset["y"]
        meta = dataset["metadata"]

        class_weights = {int(k): v for k, v in class_weights.items()}
        model_id = f"{dataset_key} | Weights {class_weights}"

        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError(f"Only one class present in target: {classes}")

        if not set(classes).issubset(class_weights.keys()):
            raise ValueError(
                f"Target classes {list(classes)} not all found in class_weights keys {list(class_weights.keys())}"
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LogisticRegression(
            random_state=42,
            solver="newton-cg",
            n_jobs=n_jobs,
            max_iter=2000,
            multi_class='multinomial',
            class_weight=class_weights
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=4)
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

        false_negatives = {
            "2 → 0": int(cm[2, 0]) if cm.shape[0] > 2 else 0,
            "2 → 1": int(cm[2, 1]) if cm.shape[0] > 2 else 0,
            "1 → 0": int(cm[1, 0]) if cm.shape[0] > 1 else 0
        }
        total_fn = sum(false_negatives.values())

        top_idx = np.argsort(np.abs(model.coef_).sum(axis=0))[::-1][:5]

        result = {
            "model": model,
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm,
            "top_coef_indices": top_idx,
            "class_weights": class_weights,
            "false_negatives": false_negatives,
            "total_false_negatives": total_fn,
            "feature_set": meta["feature_set"],
            "sampling": meta["sampling"],
            "vectorizer": meta["vectorizer"],
            "vectorizer_obj": meta.get("text_vectorizer"),
            "vectorizer_params": meta.get("vectorizer_params", {}),
            "metadata": meta
        }

        del X_train, X_test, y_train, y_test, y_pred
        gc.collect()

        return model_id, result

    def predict_with_best_model(self, sid_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the best trained model to a new DataFrame for label and probability prediction.

        Args:
            sid_df (pd.DataFrame): DataFrame to score, same structure as training input.

        Returns:
            pd.DataFrame: A copy of sid_df with appended columns:
                - 'predicted_label'
                - 'prob_green'
                - 'prob_yellow'
                - 'prob_red'
        Raises:
            ValueError: If no model has been trained or selected as best.
        """

        best = self.state.best_model
        if not best or not best.get("model"):
            raise ValueError("No trained model available in state.best_model.")
        
        model = best["model"]
        metadata = best["metadata"]
        X = self.prepare_prediction_matrix(sid_df, metadata)

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        df = sid_df.copy()
        df["predicted_label"] = y_pred
        df["prob_green"] = y_proba[:, 0] if y_proba.shape[1] > 0 else np.nan
        df["prob_yellow"] = y_proba[:, 1] if y_proba.shape[1] > 1 else np.nan
        df["prob_red"] = y_proba[:, 2] if y_proba.shape[1] > 2 else np.nan
        return df

    def load_best_model_by_key(self, model_key: str, generate_predictions: bool = True) -> Optional[pd.DataFrame]:
        """
        Reconstructs and re-trains a model from a stored model_key string.
        Sets it as self.state.best_model and self.state.best_model_key.

        Optionally generates predictions using the full dataset.

        Args:
            model_key (str): A model identifier string in the format:
                "feature_set | sampling_strategy | vectorizer_strategy | {class_weights_dict}"
            generate_predictions (bool): If True, runs predict_with_best_model(...) on sid_df.

        Returns:
            pd.DataFrame: If generate_predictions is True, returns the prediction dataframe.
        """
        parts = model_key.strip().split(" | ")
        assert len(parts) == 4, f"Model key has wrong format: {model_key}"

        feature_set = parts[0]
        sampling_strategy = parts[1]
        vectorizer_strategy = parts[2]
        class_weights_str = parts[3]

        class_weights = ast.literal_eval(class_weights_str)
        config_row = {
            "feature_set": feature_set,
            "sampling_strategy": sampling_strategy,
            "vectorizer_strategy": vectorizer_strategy
        }

        X, y, meta = self.prepare_dataset(self.state.sid_df, config_row)
        dataset = {"X": X, "y": y, "metadata": meta}

        model_id, result = self.train_one_model(
            dataset_key=f"{feature_set} | {sampling_strategy} | {vectorizer_strategy}",
            dataset=dataset,
            class_weights=class_weights,
            n_jobs=-1
        )

        # Set as best model
        self.state.best_model = result
        self.state.best_model_key = model_id
        
        # One last check
        if result.get("model") is None:
            raise ValueError(f"Model training failed. Reason: {result.get('error')}")
        
        self.conn.report("MODL", f"Rehydrated best model: {model_id}")
       
       # Optional predictions
        if generate_predictions:
            df_with_preds = self.predict_with_best_model(self.state.sid_df)
            return df_with_preds

        return None

    def merge_data(self) -> None:
        """
        Merges selected prediction columns into enriched_df on sid_key → last_sid,
        producing complete_df for downstream tasks.
        """
        enriched_df = self.state.enriched_df.copy()
        predictions_df = self.state.predictions_df.copy()

        # Columns to merge
        merge_cols = [
            "last_sid",
            "sid_key_0",
            "sid_key_1",
            "sid_key_2",
            "sid_key_3",
            "predicted_label",
            "prob_green",
            "prob_yellow",
            "prob_red",
            "all_green",
            "color_set",
            "trainable"
        ]
        predictions_df = predictions_df[merge_cols]

        # Merge
        complete_df = enriched_df.merge(
            predictions_df,
            how="left",
            left_on="sid_key",
            right_on="last_sid"
        )

        # Add human-readable color label
        complete_df["predicted_color"] = complete_df["predicted_label"].map({
            0: "Green",
            1: "Yellow",
            2: "Red"
        })

        # Store result
        self.state.complete_df = complete_df
        self.conn.report("JOIN", f"Merged predictions into enriched_df. Final shape: {complete_df.shape}")

###
# Main Pipeline Class
###
class ScoreCardPipeline:

    def __init__(self,  
                 config: ScoreCardConfig, 
                 state: ScoreCardState, 
                 conn: ConnectionManager,
                 modeler: ScoreCardModeling) -> None:

        self.config = config
        self.state = state
        self.conn = conn
        self.modeler = modeler

    def run(self) -> None:
        self._stage_1_download()
        self._stage_2_text_enrichment()
        self._stage_3_modeling_and_prediction()

    def _stage_1_download(self) -> None:
        if self.config.sql_download:
            self.download_sids()
        else:
            self.state.details_df = self.conn.load_from_es(
                index_name="scorecard_details",
                id_col="Scorecard_Detail_Note_SID"
            )

        self.conn.report("PIPE", "Completed all Downloads from SQL or ES")

    def _stage_2_text_enrichment(self) -> None:
        if self.config.enable_nlp:
            spacy.require_gpu()
            self.state.nlp = spacy.load(self.config.spacy_model)

            ScoreCardTextPrep.enrich(self.state)
            ScoreCardTextPrep.build_sid_history(self.state)

            drop_enriched = ["pre_scrub_text", "verbs", "adjectives", "noun_chunks"]
            drop_sid_df = [col for col in self.state.sid_df.columns if col.startswith("nlp_")]

            self.conn.upload_dataframe_to_es(
                df=self.state.enriched_df.drop(columns=drop_enriched),
                index_name="scorecard_enriched",
                id_col="sid_key"
            )

            self.conn.upload_dataframe_to_es(
                df=self.state.sid_df.drop(columns=drop_sid_df),
                index_name="scorecard_sid_history",
            )
        else:
            self.state.enriched_df = self.conn.load_from_es(
                index_name="scorecard_enriched",
                id_col="sid_key"
            )
            self.state.sid_df = self.conn.load_from_es(index_name="scorecard_sid_history")

        self.conn.report("PIPE", "Completed all Text Prep Steps")

    def _stage_3_modeling_and_prediction(self) -> None:
        if not self.config.build_models:
            self.state.predictions_df = self.conn.load_from_es(index_name="scorecard_predictions")
            self.modeler.merge_data()

            return

        modeler = self.modeler
        modeler.build_model_grid()

        # TEMP: Hardcoded model key for deadline
        model_key = "complete_main_words_only | no_downsample_weighted | count | {0: 0.5, 1: 1.35, 2: 1.15}"
        df_with_predictions = modeler.load_best_model_by_key(model_key)

        self.conn.report("ML", f"Best model: {self.state.best_model_key}")

        model_summary = pd.DataFrame([
            {
                "model_id": k,
                "false_negatives": v.get("total_false_negatives"),
                "y_counts": v.get("y_counts"),
                "y_pred_counts": v.get("y_pred_counts"),
                "error": v.get("error")
            }
            for k, v in self.state.models.items()
        ])
        self.conn.upload_dataframe_to_es(model_summary.fillna(""), index_name="scorecard_model_summary")

        if self.config.run_predictions:
            predictions_df = modeler.predict_with_best_model(self.state.sid_df)
            self.state.predictions_df = predictions_df

            self.conn.upload_dataframe_to_es(
                df=predictions_df,
                index_name="scorecard_predictions",
            )
            self.conn.report("ML", f"Predictions uploaded to 'scorecard_predictions'")

        self.modeler.merge_data()

    def download_sids(self) -> None:
        query = self.state.sql_query
        conn = self.state.sql_connection
        df = pd.read_sql(query, conn)

        self.state.details_df = df
        self.conn.report("SQL", f"{len(df)} rows downloaded and stored in state.details_df.")
        self.conn.upload_dataframe_to_es(df, index_name="scorecard_details")

class ScoreCardRag:
    def __init__(self, config: ScoreCardConfig, state: ScoreCardState, conn: ConnectionManager) -> None:
        self.config = config
        self.state = state
        self.conn = conn
        self.gpt = conn.gpt_client
        self.gpt_model = "gpt-4o"
        self.max_workers = os.cpu_count() // 4 or 1
        self.responses = []
        self.rag_index = config.rag_index
            
    def embed_and_index_notes(self) -> None:
        """
        Embeds Scorecard_Note and indexes the entire complete_df into the RAG index.
        Includes token stats, created_at timestamp, and filters out archived rows if needed.
        """
        es = self.conn.es_client
        df = self.state.complete_df.copy()

        # Optional: filter archived notes (comment this out if you want everything)
        # df = df[df["Archive_Indicator"] != "Y"].copy()

        # Step 1: Use Scorecard_Note as embedding input
        df["text_for_embedding"] = df["Scorecard_Note"].fillna("").astype(str)

        # A little date flavoring
        df["Note_YearMonth"] = df["Note_Year"].astype(int) * 100 + df["Note_Month"].astype(int)
        
        # Convert Report_Year and Report_Month to a datetime (1st of the month)
        df["Report_Date"] = pd.to_datetime(df["Report_Year"].astype(str) + "-" + df["Report_Month"].astype(str).str.zfill(2) + "-01")

        # Step 2: Load embedding model and tokenizer
        embedding_model = self.conn.state.embedding_model
        tokenizer = tiktoken.get_encoding(self.config.tokenizer_name)

        # Step 3: Embed
        self.conn.report("EMBD", f"Encoding {len(df)} notes for embedding...")
        embeddings = embedding_model.encode(
            df["text_for_embedding"].tolist(),
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            device="cuda"
        )
        df["embedding"] = embeddings.tolist()

        # Step 4: Token count + metadata
        df["tokens"] = df["text_for_embedding"].apply(lambda text: tokenizer.encode(text))
        df["token_count"] = df["tokens"].apply(len)
        df["embedding_version"] = self.config.embedding_model_name
        df["gpt_justification"] = ""
        df["created_at"] = datetime.utcnow().isoformat()
        
        # Use sid_key as the ES _id to ensure each note has a stable, unique anchor ID
        df["_id"] = df["sid_key"].astype(str)

        # Optional: token stats
        token_stats = df["token_count"].describe().to_dict()
        self.conn.report("EMBD", f"Token count stats: {json.dumps(token_stats, indent=2)}")

        # Step 5: Reset the RAG index
        if es.indices.exists(index=self.rag_index):
            es.indices.delete(index=self.rag_index)
            self.conn.report("ES", f"Deleted existing index '{self.rag_index}'")

        mapping = {
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.config.embedding_vector_dim,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "tokens": {"type": "integer"},
                    "token_count": {"type": "integer"},
                    "embedding_version": {"type": "keyword"},
                    "gpt_justification": {"type": "text"},
                    "created_at": {"type": "date"},
                    "Note_YearMonth": {"type": "integer"},
                    "SID": {"type": "integer"},
                    "Note_Year": {"type": "integer"},
                    "Note_Month": {"type": "integer"},
                    "Report_Date": {"type": "date"}
                }
            }
        }

        es.indices.create(index=self.rag_index, body=mapping)
        self.conn.report("ES", f"Created index '{self.rag_index}' with vector mapping")

        # Step 6: Bulk index everything
        actions = []
        for _, row in df.iterrows():
            actions.append({
                "_index": self.rag_index,
                "_id": row["_id"],
                "_source": self.conn.serialize_row_for_es(row)
            })

        success, errors = bulk(es, actions, raise_on_error=False)
        self.conn.report("ES", f"Indexed {success} documents to '{self.rag_index}'")

        if errors:
            self.conn.report("ES", f"{len(errors)} document(s) failed to index.")
            for err in errors[:3]:
                print(json.dumps(err, indent=2))
            raise RuntimeError(f"Errors occurred indexing to '{self.rag_index}'")

    def similar_notes_same_sid(self, anchor_sid_key: str, top_k: int = 5) -> list[dict]:
        """
        Returns top-k similar *past* notes from the same SID, based on sid_key-encoded date.
        Uses cosine similarity over embedded vectors stored in Elasticsearch.
        """
        es = self.conn.es_client

        # --- Parse sid_key format: SID.YYYY.MM.NNNNNN ---
        sid_parts = anchor_sid_key.split(".")
        assert len(sid_parts) == 4, f"Invalid sid_key format: {anchor_sid_key}"

        anchor_sid = int(sid_parts[0])
        anchor_year = int(sid_parts[1])
        anchor_month = int(sid_parts[2])
        anchor_date_val = anchor_year * 100 + anchor_month

        # --- Get embedding from Elasticsearch (since we use sid_key as _id) ---
        query_vector = self._get_embedding(anchor_sid_key)

        # --- Build RAG similarity + filter query ---
        query = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"SID": anchor_sid}},  # same SID
                                {
                                    "script": {
                                        "script": {
                                            "source": (
                                                "(doc['Note_Year'].value * 100 + doc['Note_Month'].value) "
                                                "< params.date"
                                            ),
                                            "params": {"date": anchor_date_val}
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }

        # --- Execute search ---
        response = es.search(index=self.rag_index, body=query)
        hits = response.get("hits", {}).get("hits", [])

        if not hits:
            self.conn.report("RAG", f"No prior notes found for SID {anchor_sid} before {anchor_year}-{anchor_month}")

        return [hit["_source"] for hit in hits]

    def get_vendor_trouble_notes(self, anchor_sid_key: str, top_k: int = 5) -> list[dict]:
        df = self.state.complete_df

        # Step 1: Parse the anchor sid_key
        sid_parts = anchor_sid_key.split(".")
        assert len(sid_parts) == 4, f"Invalid sid_key format: {anchor_sid_key}"
        anchor_sid = int(sid_parts[0])

        # Step 2: Get vendor ID from anchor row
        anchor_row = df[df["sid_key"] == anchor_sid_key]
        if anchor_row.empty:
            raise ValueError(f"No row found with sid_key={anchor_sid_key}")
        vendor_id = anchor_row.iloc[0]["LM_Vendor_ID"]

        # Step 3: Filter for other notes from same vendor, different SID, non-green
        filtered = df[
            (df["LM_Vendor_ID"] == vendor_id) &
            (df["SID"] != anchor_sid) &
            (df["Overall"] != "G")
        ].copy()

        # Step 4: Add a sort key for recency
        filtered["Note_YearMonth"] = (
            filtered["Note_Year"].astype(int) * 100 + filtered["Note_Month"].astype(int)
        )
        filtered.sort_values(by="Note_YearMonth", ascending=False, inplace=True)

        # Step 5: Group by SID, take top note per SID
        top_notes = (
            filtered.groupby("SID", as_index=False)
            .head(1)
            .sort_values(by="Note_YearMonth", ascending=False)
            .head(top_k)
        )

        return top_notes[["sid_key", "SID", "Scorecard_Note"]].rename(
            columns={"sid_key": "trouble_note_key"}
        ).to_dict(orient="records")

    def retrieve_augmented_history(self, anchor_sid_key: str, top_k: int = 5) -> dict:
        """
        Builds a GPT-friendly payload for a given sid_key.
        Includes:
            - Static metadata
            - Explicit note history (sid_key_0, sid_key_1, ...)
            - Similar prior notes from same SID (excluding duplicates)
            - Vendor trouble notes (other SIDs, same vendor, non-green)
        """
        df = self.state.complete_df
        anchor_row = df[df["sid_key"] == anchor_sid_key]
        if anchor_row.empty:
            raise ValueError(f"No row found with sid_key={anchor_sid_key}")

        row = anchor_row.iloc[0]

        # --- Static metadata ---
        metadata_fields = [
            "SID", "LM_Vendor_ID", "predicted_label", "prob_green", "prob_yellow", "prob_red",
            "all_green", "color_set", "trainable", "predicted_color", "Program_Name"
        ]
        metadata = {k: row[k] for k in metadata_fields if k in row}

        anchor_sid = row["SID"]

        # --- Explicit note history from sid_key_0, sid_key_1, ...
        explicit_keys = [
            row.get(f"sid_key_{i}")
            for i in range(10)
            if f"sid_key_{i}" in row and pd.notnull(row[f"sid_key_{i}"])
        ]
        explicit_keys_set = set(explicit_keys)

        explicit_notes = (
            df[df["sid_key"].isin(explicit_keys)][["sid_key", "Scorecard_Note"]]
            .rename(columns={"sid_key": "document_key"})
            .to_dict(orient="records")
        )

        # --- Similar notes via vector search (same SID, earlier date) ---
        similar_notes = self.similar_notes_same_sid(anchor_sid_key, top_k=top_k)
        similar_cleaned = [
            {
                "document_key": note.get("sid_key", "unknown"),
                "Scorecard_Note": note.get("Scorecard_Note", "")
            }
            for note in similar_notes
            if note.get("sid_key") not in explicit_keys_set
        ]

        sid_key_seen = explicit_keys_set.union({n["document_key"] for n in similar_cleaned})

        # --- Vendor trouble notes (structured search only, no ES) ---
        vendor_troubles_raw = self.get_vendor_trouble_notes(anchor_sid_key, top_k=top_k)
        vendor_troubles = [
            note for note in vendor_troubles_raw
            if note["trouble_note_key"] not in sid_key_seen
        ]

        # --- Final payload ---
        full_history = sorted(explicit_notes + similar_cleaned, key=lambda x: x["document_key"])
        vendor_troubles = sorted(vendor_troubles, key=lambda x: x["trouble_note_key"])

        return {
            "metadata": metadata,
            "note_history": full_history,
            "vendor_troubles": vendor_troubles
        }

    def _format_note_list(self, notes: list[dict], key_name: str = "document_key") -> str:
        """
        Converts a list of notes into a formatted string block for GPT prompt.

        Args:
            notes: List of dicts with keys like 'Scorecard_Note' and an ID key.
            key_name: The key used to extract the document or trouble note ID.

        Returns:
            A formatted string for inclusion in the GPT prompt.
        """
        lines = []
        for note in notes:
            key = note.get(key_name, "unknown")
            sid = key.split(".")[0] if "." in key else "?"
            note_id = key.split(".")[-1]
            summary = note.get("Scorecard_Note", "").strip()
            lines.append(f"[{key}] (SID {sid}, Note {note_id})\n{summary}\n")
        return "\n".join(lines)

    def _confidence_from_label(self, label: int, metadata: dict) -> float:
        """
        Retrieves the prediction confidence (as percentage) for a given label using metadata.

        Args:
            label: The predicted label (0 = Green, 1 = Yellow, 2 = Red).
            metadata: A dictionary with keys 'prob_green', 'prob_yellow', 'prob_red'.

        Returns:
            The confidence value as a float percentage.
        """
        confidence_map = {
            0: metadata.get("prob_green", 0),
            1: metadata.get("prob_yellow", 0),
            2: metadata.get("prob_red", 0),
        }
        return confidence_map.get(label, 0.0) * 100

    def generate_justifications(self, anchor_sid_key: str, printer: bool = False ) -> None:
        """
        Generates a GPT-based justification for a given sid_key and updates Elasticsearch.
        Uses GPT system prompt from self.state.gpt_prompt and metadata/history from retrieve_augmented_history().
        """
        es = self.conn.es_client
        gpt = self.conn.gpt_client
        prompt_template = self.state.gpt_prompt

        # Step 1: Retrieve structured history and metadata
        data = self.retrieve_augmented_history(anchor_sid_key)
        metadata = data["metadata"]
        note_history = data["note_history"]
        vendor_troubles = data["vendor_troubles"]

        # Step 2: Format notes
        recent_notes_block = self._format_note_list(note_history, key_name="document_key")
        vendor_troubles_block = self._format_note_list(vendor_troubles, key_name="trouble_note_key")

        # Step 3: Compute prediction confidence
        label = metadata.get("predicted_label", -1)
        confidence = self._confidence_from_label(label, metadata)

        prompt = prompt_template.format(
            predicted_color=metadata.get("predicted_color", "Unknown"),
            confidence=confidence,
            SID=metadata.get("SID", "Unknown"),
            LM_Vendor_ID=metadata.get("LM_Vendor_ID", "Unknown"),
            Program_Name=metadata.get("Program_Name", "Unknown"),
            note_history=recent_notes_block,               
            vendor_troubles=vendor_troubles_block          
        )

        # Step 5: Call GPT
        response = gpt.chat.completions.create(
            model=self.gpt_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        justification = response.choices[0].message.content.strip()

        # Step 6: Save to Elasticsearch
        es.update(
            index=self.rag_index,
            id=anchor_sid_key,
            body={"doc": {"justification": justification}},
        )

        if printer:
            self.conn.report("AI", f"Justification saved for {anchor_sid_key}. Here is the first part:\n {justification}")

    def _get_embedding(self, sid_key: str) -> list[float]:
        """
        Fetches embedding vector directly from Elasticsearch using sid_key as _id.
        """
        es = self.conn.es_client
        try:
            res = es.get(index=self.rag_index, id=sid_key)
            return res["_source"]["embedding"]
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve embedding for sid_key={sid_key}: {e}")

    def run_gpt_justification_pass(self, limit: Optional[int] = None, max_attempts: int = 6, backoff: float = 1.5) -> None:
        """
        Runs GPT justification pass across the complete_df using multithreading with retries and backoff.

        Args:
            limit: Number of rows to process. If None, process all.
            max_attempts: Max number of retry attempts per sid_key.
            backoff: Base sleep multiplier (in seconds) between retries.
        """
        df = self.state.complete_df.copy()

        if limit is not None:
            df = df.head(limit)

        sid_keys = df["sid_key"].tolist()
        results = [None] * len(sid_keys)

        def process(i: int, sid_key: str):
            for attempt in range(1, max_attempts + 1):
                try:
                    time.sleep(attempt * attempt * backoff)  # staggered backoff
                    self.generate_justifications(sid_key)
                    self.conn.report("GPT", f"[{i}] {sid_key} succeeded (attempt {attempt})")
                    results[i] = "OK"
                    return
                except Exception as e:
                    self.conn.report("GPT", f"[{i}] {sid_key} failed (attempt {attempt}): {e}")
            results[i] = f"FAILED after {max_attempts} attempts"

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process, i, sid_key): i
                for i, sid_key in enumerate(sid_keys)
            }
            for future in as_completed(futures):
                _ = future.result()

        success_count = results.count("OK")
        fail_count = len(results) - success_count
        self.conn.report("DONE", f"Justification pass complete: {success_count} succeeded, {fail_count} failed.")

