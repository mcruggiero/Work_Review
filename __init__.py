# ========================================================================
# SCORECARD PACKAGE
# Multi-horizon prediction system for subcontract scorecards
# ========================================================================

"""
ScoreCard ML Pipeline Package

This package provides a complete ML pipeline for predicting subcontract
scorecard outcomes (Green/Yellow/Red) with multi-horizon support.

Main Components:
    - ScoreCardConfig: Configuration dataclass
    - ScoreCardState: State management dataclass
    - ConnectionManager: Database and API connections
    - ScoreCardTextPrep: NLP text preprocessing
    - ScoreCardModeling: ML model training and prediction
    - ScoreCardRag: RAG embeddings and GPT justifications
    - ScoreCardPipeline: Main pipeline orchestrator
    - upload: SQL Server upload pipeline for model results
    - reports: Dashboard visualizations and flagged prediction analysis

Horizons:
    - H1: Predict next card (1 step ahead)
    - H2: Predict card after next (2 steps ahead)

Usage:
    from scorecard import run_pipeline

    state, pipeline, rag = run_pipeline(
        sql_download=True,
        enable_nlp=True,
        build_models=True,
        run_predictions=True,
        build_rag=True,
    )

Or import individual components:
    from scorecard import (
        ScoreCardConfig,
        ScoreCardState,
        ConnectionManager,
        ScoreCardModeling,
        ScoreCardPipeline,
        ScoreCardRag,
        Horizon,
        SUPPORTED_HORIZONS,
    )

For uploading results to SQL:
    from scorecard import upload_predictions_to_sql

For generating reports:
    from scorecard import plot_prediction_dashboard, generate_flagged_report

For model validation:
    from scorecard import (
        compute_baseline_metrics,      # Compare to naive baselines
        plot_calibration_curves,       # Probability calibration
        plot_precision_recall_curves,  # PR curves per class
        plot_error_analysis,           # Deep error analysis
        plot_feature_importance,       # Feature coefficients
        generate_model_validation_report,  # Full validation suite
    )
"""

from .config import (
    ScoreCardConfig,
    ScoreCardState,
    Horizon,
    SUPPORTED_HORIZONS,
    horizon_offset,
)

from .connections import ConnectionManager

from .text_prep import ScoreCardTextPrep

from .modeling import ScoreCardModeling

from .rag import ScoreCardRag

from .pipeline import ScoreCardPipeline

from .main import run_pipeline, main

from .upload import (
    get_upload_engine,
    upload_predictions_to_sql,
    build_upload_table,
    get_schema_info,
    preflight_report,
    trim_df_to_schema,
)

from .pickler import (
    save_state,
    load_state,
    save_dataframes_only,
    load_dataframes_only,
)

from .reports import (
    # Core reporting
    enrich_for_reporting,
    generate_summary_tables,
    plot_prediction_dashboard,
    display_one_note,
    display_flagged_notes,
    generate_flagged_report,
    save_notes_html,
    # Model validation - Baseline comparison
    compute_baseline_metrics,
    plot_baseline_comparison,
    # Model validation - Calibration
    compute_calibration_metrics,
    plot_calibration_curves,
    # Model validation - Precision-Recall
    plot_precision_recall_curves,
    # Model validation - Temporal analysis
    temporal_train_test_split,
    analyze_temporal_performance,
    plot_temporal_performance,
    # Model validation - Error analysis
    analyze_errors,
    plot_error_analysis,
    # Model validation - Feature importance
    extract_feature_importance,
    plot_feature_importance,
    generate_word_clouds,
    # Comprehensive validation report
    generate_model_validation_report,
)


__all__ = [
    # Config
    "ScoreCardConfig",
    # State persistence
    "save_state",
    "load_state",
    "save_dataframes_only",
    "load_dataframes_only",
    "ScoreCardState",
    "Horizon",
    "SUPPORTED_HORIZONS",
    "horizon_offset",
    # Connections
    "ConnectionManager",
    # Text Prep
    "ScoreCardTextPrep",
    # Modeling
    "ScoreCardModeling",
    # RAG
    "ScoreCardRag",
    # Pipeline
    "ScoreCardPipeline",
    # Entry points
    "run_pipeline",
    "main",
    # Upload
    "get_upload_engine",
    "upload_predictions_to_sql",
    "build_upload_table",
    "get_schema_info",
    "preflight_report",
    "trim_df_to_schema",
    # Reports - Core
    "enrich_for_reporting",
    "generate_summary_tables",
    "plot_prediction_dashboard",
    "display_one_note",
    "display_flagged_notes",
    "generate_flagged_report",
    "save_notes_html",
    # Reports - Model Validation
    "compute_baseline_metrics",
    "plot_baseline_comparison",
    "compute_calibration_metrics",
    "plot_calibration_curves",
    "plot_precision_recall_curves",
    "temporal_train_test_split",
    "analyze_temporal_performance",
    "plot_temporal_performance",
    "analyze_errors",
    "plot_error_analysis",
    "extract_feature_importance",
    "plot_feature_importance",
    "generate_word_clouds",
    "generate_model_validation_report",
]

__version__ = "2.0.0"
