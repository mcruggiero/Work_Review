# ========================================================================
# SCORECARD PICKLER MODULE
# Utilities for saving and loading pipeline state across notebooks
# ========================================================================

"""
ScoreCard State Persistence

This module provides utilities to save and load pipeline state objects
between Jupyter notebooks. Useful when you want to:
  - Run the expensive pipeline once and analyze results in separate notebooks
  - Resume work after a kernel restart
  - Share state between team members

Usage (in source notebook):
    from scorecard.pickler import save_state
    save_state(state, pipeline, rag, config, conn, path="./pipeline_state.pkl")

Usage (in target notebook):
    from scorecard.pickler import load_state
    state, pipeline, rag, config, conn = load_state(path="./pipeline_state.pkl")

Note: Some objects (spaCy models, connections, embedding models) cannot be
pickled directly. These are handled specially:
  - DataFrames and model results are fully preserved
  - Connections are re-established on load
  - Large ML objects (nlp, embedding_model) are reloaded from disk
"""

import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Tuple
from dataclasses import asdict

import pandas as pd


# ========================================================================
# SAVE STATE
# ========================================================================

def save_state(
    state,
    pipeline=None,
    rag=None,
    config=None,
    conn=None,
    path: str = "./pipeline_state.pkl",
    include_models: bool = True,
    verbose: bool = True,
) -> str:
    """
    Save pipeline state to disk for later resumption.

    Parameters
    ----------
    state : ScoreCardState
        The main state object containing dataframes and model results
    pipeline : ScoreCardPipeline, optional
        The pipeline object (models stored in state)
    rag : ScoreCardRag, optional
        The RAG object (embeddings indexed in ES)
    config : ScoreCardConfig, optional
        Configuration object (will use state.config if not provided)
    conn : ConnectionManager, optional
        Connection manager (will be re-established on load)
    path : str
        Output file path (default: ./pipeline_state.pkl)
    include_models : bool
        Whether to include trained sklearn models (default: True)
    verbose : bool
        Print progress messages (default: True)

    Returns
    -------
    str
        Path to saved file
    """
    if verbose:
        print("=" * 60)
        print("SAVING PIPELINE STATE")
        print("=" * 60)

    # Use state's config if not provided
    if config is None:
        config = state.config

    # Build serializable state dict
    state_dict = {
        "_metadata": {
            "saved_at": datetime.now().isoformat(),
            "version": "1.0",
            "include_models": include_models,
        },
        # Configuration (as dict for easier serialization)
        "config_dict": _config_to_dict(config),
        # DataFrames - the core data we want to preserve
        "dataframes": {
            "raw_df": state.raw_df,
            "details_df": state.details_df,
            "enriched_df": state.enriched_df,
            "sid_df": state.sid_df,
            "predicted_sids": state.predicted_sids,
            "complete_dataset": state.complete_dataset,
            "complete_df": state.complete_df,
            "predictions_df": state.predictions_df,
            "final_df": state.final_df,
        },
        # Model results by horizon
        "model_results": {
            "models_by_horizon": state.models_by_horizon if include_models else {},
            "best_model_by_horizon": state.best_model_by_horizon if include_models else {},
            "best_model_key_by_horizon": state.best_model_key_by_horizon,
            "predictions_df_by_horizon": state.predictions_df_by_horizon,
        },
        # Single-horizon model results (backward compat)
        "legacy_models": {
            "models": state.models if include_models else {},
            "best_model": state.best_model if include_models else None,
            "best_model_key": state.best_model_key,
            "model_grid_df": state.model_grid_df,
            "prepared_datasets": state.prepared_datasets,
        },
        # Model matrix configs
        "model_matrices": {
            "feature_matrix": state.feature_matrix,
            "sampling_matrix": state.sampling_matrix,
            "vectorization_matrix": state.vectorization_matrix,
            "model_weights": state.model_weights,
            "vectorizer_params": state.vectorizer_params,
        },
        # Prompts and queries
        "prompts": {
            "sql_query": state.sql_query,
            "gpt_prompt": state.gpt_prompt,
        },
    }

    # Save vectorizer if available (sklearn vectorizers are picklable)
    if include_models and state.vectorizer is not None:
        try:
            state_dict["vectorizer"] = state.vectorizer
            if verbose:
                print("[SAVE] \tVectorizer included")
        except Exception as e:
            warnings.warn(f"Could not pickle vectorizer: {e}")
            state_dict["vectorizer"] = None

    # Log what we're saving
    if verbose:
        print(f"[SAVE] \tConfig: {len(state_dict['config_dict'])} settings")
        df_count = sum(1 for v in state_dict["dataframes"].values() if v is not None)
        print(f"[SAVE] \tDataFrames: {df_count} non-null")
        for name, df in state_dict["dataframes"].items():
            if df is not None:
                print(f"[SAVE] \t  - {name}: {df.shape}")
        print(f"[SAVE] \tHorizons: {list(state_dict['model_results']['best_model_key_by_horizon'].keys())}")
        if include_models:
            model_count = sum(len(v) for v in state_dict["model_results"]["models_by_horizon"].values())
            print(f"[SAVE] \tModels: {model_count} total")

    # Write to disk
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = path.stat().st_size / (1024 * 1024)
    if verbose:
        print(f"[SAVE] \tSaved to: {path}")
        print(f"[SAVE] \tFile size: {file_size_mb:.1f} MB")
        print("=" * 60)

    return str(path)


def _config_to_dict(config) -> dict:
    """Convert ScoreCardConfig to a plain dict."""
    return {
        "spacy_model": config.spacy_model,
        "json_path": config.json_path,
        "model_matrix_json": config.model_matrix_json,
        "vectorizer_strategy": config.vectorizer_strategy,
        "sql_driver_path": config.sql_driver_path,
        "sql_server": config.sql_server,
        "sql_database": config.sql_database,
        "sql_uid": config.sql_uid,
        "sql_pwd": config.sql_pwd,
        "sql_query_file": config.sql_query_file,
        "require_gpu": config.require_gpu,
        "sql_download": config.sql_download,
        "enable_nlp": config.enable_nlp,
        "build_models": config.build_models,
        "run_predictions": config.run_predictions,
        "build_rag": config.build_rag,
        "es_host": config.es_host,
        "elastic_index": config.elastic_index,
        "tokenizer_name": config.tokenizer_name,
        "embedding_model_name": config.embedding_model_name,
        "embedding_vector_dim": config.embedding_vector_dim,
        "rag_index": config.rag_index,
        "batch_size": config.batch_size,
        "training_length": config.training_length,
        "default_model_key_h1": config.default_model_key_h1,
        "default_model_key_h2": config.default_model_key_h2,
        "gpt_prompt_location": config.gpt_prompt_location,
        "gpt_base_url": config.gpt_base_url,
        "log_index_name": config.log_index_name,
        "log_file_path": config.log_file_path,
    }


# ========================================================================
# LOAD STATE
# ========================================================================

def load_state(
    path: str = "./pipeline_state.pkl",
    reconnect: bool = True,
    reload_nlp: bool = False,
    reload_embeddings: bool = False,
    verbose: bool = True,
) -> Tuple[Any, Optional[Any], Optional[Any], Any, Optional[Any]]:
    """
    Load pipeline state from disk.

    Parameters
    ----------
    path : str
        Path to saved state file
    reconnect : bool
        Re-establish database connections (default: True)
    reload_nlp : bool
        Reload spaCy model (default: False - not needed for analysis)
    reload_embeddings : bool
        Reload embedding model (default: False - not needed unless using RAG)
    verbose : bool
        Print progress messages (default: True)

    Returns
    -------
    tuple
        (state, pipeline, rag, config, conn)
        Note: pipeline and rag may be None if not needed for analysis
    """
    if verbose:
        print("=" * 60)
        print("LOADING PIPELINE STATE")
        print("=" * 60)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")

    file_size_mb = path.stat().st_size / (1024 * 1024)
    if verbose:
        print(f"[LOAD] \tFile: {path}")
        print(f"[LOAD] \tSize: {file_size_mb:.1f} MB")

    with open(path, "rb") as f:
        state_dict = pickle.load(f)

    # Check metadata
    meta = state_dict.get("_metadata", {})
    if verbose:
        print(f"[LOAD] \tSaved at: {meta.get('saved_at', 'unknown')}")
        print(f"[LOAD] \tVersion: {meta.get('version', 'unknown')}")
        print(f"[LOAD] \tIncludes models: {meta.get('include_models', False)}")

    # Reconstruct config
    from scorecard import ScoreCardConfig
    config_dict = state_dict["config_dict"]
    config = ScoreCardConfig(**config_dict)
    if verbose:
        print(f"[LOAD] \tConfig reconstructed")

    # Create a lightweight state object (skip heavy initialization)
    state = _create_lightweight_state(state_dict, config, reload_nlp, verbose)

    # Re-establish connections if requested
    conn = None
    if reconnect:
        try:
            from scorecard import ConnectionManager
            conn = ConnectionManager(config=config, state=state)
            if verbose:
                print(f"[LOAD] \tConnections re-established")
        except Exception as e:
            warnings.warn(f"Could not reconnect: {e}")
            if verbose:
                print(f"[LOAD] \tConnections failed: {e}")

    # RAG object (lightweight - just needs config and conn for GPT calls)
    rag = None
    if reconnect and reload_embeddings:
        try:
            from scorecard import ScoreCardRag
            rag = ScoreCardRag(config=config, state=state, conn=conn)
            if verbose:
                print(f"[LOAD] \tRAG object created")
        except Exception as e:
            warnings.warn(f"Could not create RAG: {e}")

    # Pipeline object is not needed for analysis (models are in state)
    pipeline = None

    if verbose:
        print("=" * 60)
        print("STATE LOADED SUCCESSFULLY")
        print("=" * 60)
        _print_state_summary(state)

    return state, pipeline, rag, config, conn


def _create_lightweight_state(state_dict: dict, config, reload_nlp: bool, verbose: bool):
    """
    Create a ScoreCardState without heavy initialization.

    This bypasses __post_init__ to avoid loading spaCy, etc.
    """
    from scorecard import ScoreCardState

    # Create state object but skip __post_init__
    state = object.__new__(ScoreCardState)

    # Set config
    state.config = config

    # Restore DataFrames
    dfs = state_dict.get("dataframes", {})
    state.raw_df = dfs.get("raw_df")
    state.details_df = dfs.get("details_df")
    state.enriched_df = dfs.get("enriched_df")
    state.sid_df = dfs.get("sid_df")
    state.predicted_sids = dfs.get("predicted_sids")
    state.complete_dataset = dfs.get("complete_dataset")
    state.complete_df = dfs.get("complete_df")
    state.predictions_df = dfs.get("predictions_df")
    state.final_df = dfs.get("final_df")

    if verbose:
        df_count = sum(1 for v in [
            state.raw_df, state.details_df, state.enriched_df,
            state.sid_df, state.predictions_df
        ] if v is not None)
        print(f"[LOAD] \tDataFrames restored: {df_count}")

    # Restore model results
    mr = state_dict.get("model_results", {})
    state.models_by_horizon = mr.get("models_by_horizon", {})
    state.best_model_by_horizon = mr.get("best_model_by_horizon", {})
    state.best_model_key_by_horizon = mr.get("best_model_key_by_horizon", {})
    state.predictions_df_by_horizon = mr.get("predictions_df_by_horizon", {})

    if verbose:
        print(f"[LOAD] \tHorizons: {list(state.best_model_key_by_horizon.keys())}")

    # Restore legacy single-horizon models
    lm = state_dict.get("legacy_models", {})
    state.models = lm.get("models", {})
    state.best_model = lm.get("best_model")
    state.best_model_key = lm.get("best_model_key")
    state.model_grid_df = lm.get("model_grid_df")
    state.prepared_datasets = lm.get("prepared_datasets")

    # Restore model matrices
    mm = state_dict.get("model_matrices", {})
    state.feature_matrix = mm.get("feature_matrix")
    state.sampling_matrix = mm.get("sampling_matrix")
    state.vectorization_matrix = mm.get("vectorization_matrix")
    state.model_weights = mm.get("model_weights")
    state.vectorizer_params = mm.get("vectorizer_params")

    # Restore prompts
    prompts = state_dict.get("prompts", {})
    state.sql_query = prompts.get("sql_query")
    state.gpt_prompt = prompts.get("gpt_prompt")

    # Restore vectorizer if available
    state.vectorizer = state_dict.get("vectorizer")
    if state.vectorizer is not None and verbose:
        print(f"[LOAD] \tVectorizer restored")

    # Initialize empty placeholders for non-serializable objects
    state.nlp = None
    state.embedding_model = None
    state.embedding_dim = config.embedding_vector_dim
    state.tokenizer = None
    state.X_matrix = None
    state.y_vector = None
    state.sql_connection = None
    state.es_conn = None
    state.gpt_client = None

    # Optionally reload spaCy
    if reload_nlp:
        import spacy
        if verbose:
            print(f"[LOAD] \tLoading spaCy model: {config.spacy_model}")
        state.nlp = spacy.load(config.spacy_model)

    return state


def _print_state_summary(state):
    """Print a summary of what's in the loaded state."""
    print("\nState Summary:")
    print("-" * 40)

    # DataFrames
    dfs = [
        ("enriched_df", state.enriched_df),
        ("predictions_df", state.predictions_df),
        ("complete_df", state.complete_df),
    ]
    for name, df in dfs:
        if df is not None:
            print(f"  {name}: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # Horizons
    for h, key in state.best_model_key_by_horizon.items():
        print(f"  H{h} model key: {key[:50]}...")

    # Predictions by horizon
    for h, df in state.predictions_df_by_horizon.items():
        if df is not None:
            print(f"  H{h} predictions: {df.shape[0]:,} rows")


# ========================================================================
# CONVENIENCE FUNCTIONS
# ========================================================================

def save_dataframes_only(
    state,
    path: str = "./pipeline_dataframes.pkl",
    verbose: bool = True,
) -> str:
    """
    Save only the DataFrames from state (smaller file, faster save/load).

    Useful when you only need the data for analysis, not the models.
    """
    if verbose:
        print("Saving DataFrames only...")

    dfs = {
        "enriched_df": state.enriched_df,
        "predictions_df": state.predictions_df,
        "predictions_df_by_horizon": state.predictions_df_by_horizon,
        "complete_df": state.complete_df,
        "sid_df": state.sid_df,
    }

    path = Path(path)
    with open(path, "wb") as f:
        pickle.dump(dfs, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Saved to {path} ({size_mb:.1f} MB)")

    return str(path)


def load_dataframes_only(path: str = "./pipeline_dataframes.pkl") -> dict:
    """Load DataFrames saved with save_dataframes_only()."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ========================================================================
# EXPORTS
# ========================================================================

__all__ = [
    "save_state",
    "load_state",
    "save_dataframes_only",
    "load_dataframes_only",
]
