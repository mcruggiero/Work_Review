# ScoreCard Multi-Horizon Prediction Pipeline

A machine learning pipeline for predicting supplier scorecard ratings (Green/Yellow/Red) with multi-horizon support. The system predicts both the **next card (H1)** and the **card after next (H2)** using separate models trained directly from historical data.

## Project Structure

```
carry_over/
├── README.md                 # This file
├── notebooks/
│   ├── 15_Modular.ipynb     # Modular component development
│   ├── 17_SQL_Upload.ipynb  # SQL data upload utilities
│   ├── 17a_UploadRebuild.ipynb
│   ├── 18_ScoreCardDemo.ipynb  # Main demonstration notebook
│   └── global_summary_dashboard.png
└── src/
    ├── __init__.py
    ├── scorecard.py          # Main pipeline module (multi-horizon enabled)
    ├── scorecard-Copy1.py    # Backup of original single-horizon version
    ├── GPT_Prompt.txt        # System prompt template for GPT justifications
    ├── sql_query.txt         # SQL query for data extraction
    └── requirements.txt      # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r src/requirements.txt
```

Key dependencies:
- `pandas`, `numpy`, `scikit-learn` - Data processing and ML
- `spacy` (with `en_core_web_trf`) - NLP text processing
- `elasticsearch` - Data storage and vector search
- `sentence-transformers` - Text embeddings for RAG
- `openai` - GPT API for justification generation
- `torch` - GPU acceleration

### 2. Download SpaCy Model

```bash
python -m spacy download en_core_web_trf
```

### 3. Configure the Pipeline

Edit the `ScoreCardConfig` dataclass in `src/scorecard.py` or pass custom values:

```python
from scorecard import ScoreCardConfig, ScoreCardState, ConnectionManager, ScoreCardModeling, ScoreCardPipeline

# Default configuration
config = ScoreCardConfig(
    sql_download=True,        # Set False to load from Elasticsearch
    enable_nlp=True,          # Run NLP enrichment
    build_models=True,        # Train models
    run_predictions=True,     # Generate predictions
    build_rag=False,          # Set True for GPT justifications
)

# Initialize state and connections
state = ScoreCardState(config=config)
conn = ConnectionManager(config, state)
modeler = ScoreCardModeling(config, state, conn)

# Create and run pipeline
pipeline = ScoreCardPipeline(config, state, conn, modeler)
pipeline.run()
```

### 4. Run via Notebook

The easiest way to run the pipeline is via the demonstration notebook:

```bash
jupyter notebook notebooks/18_ScoreCardDemo.ipynb
```

## Multi-Horizon Prediction

The pipeline now supports two prediction horizons:

| Horizon | Description | Target | Minimum Notes |
|---------|-------------|--------|---------------|
| **H1** | Next card prediction | 1 step ahead | 5 notes |
| **H2** | Card after next | 2 steps ahead | 6 notes |

### Key Design Decisions

1. **No Recursive Predictions**: H2 is trained directly from historical data, NOT conditioned on H1 predictions. This ensures temporal correctness.

2. **Backward Compatibility**: H1 uses original column names (`trainable`, `target`, `predicted_label`). H2 uses `_h2` suffix (`trainable_h2`, `target_h2`, `predicted_label_h2`).

3. **Separate Models**: Each horizon has its own trained model with independent evaluation metrics and model selection.

4. **Extensibility**: Adding H3 requires only adding `H3 = 3` to the `Horizon` enum and `SUPPORTED_HORIZONS` list.

### Output Columns

After running the pipeline, `state.complete_df` contains:

**H1 Columns (original names):**
- `trainable` - Whether row was used for H1 training
- `target` - H1 target label (0=Green, 1=Yellow, 2=Red)
- `predicted_label` - H1 predicted label
- `prob_green`, `prob_yellow`, `prob_red` - H1 probabilities
- `predicted_color` - Human-readable H1 prediction

**H2 Columns (with _h2 suffix):**
- `trainable_h2` - Whether row was used for H2 training
- `target_h2` - H2 target label
- `predicted_label_h2` - H2 predicted label
- `prob_green_h2`, `prob_yellow_h2`, `prob_red_h2` - H2 probabilities
- `predicted_color_h2` - Human-readable H2 prediction

## Pipeline Stages

### Stage 1: Data Download
- Downloads supplier notes from SQL Server or loads from Elasticsearch
- Output: `state.details_df`

### Stage 2: Text Enrichment
- Cleans HTML, extracts NLP features (verbs, adjectives, noun chunks)
- Builds sliding windows with horizon-specific trainability
- Output: `state.enriched_df`, `state.sid_df`

### Stage 3: Modeling & Prediction
- Builds model grid (feature sets × sampling × vectorizers × weights)
- Trains separate models for H1 and H2
- Selects best model per horizon by minimizing false negatives
- Generates predictions for all data
- Output: `state.predictions_df`, `state.predictions_df_by_horizon`, `state.complete_df`

### Stage 4: RAG & GPT Justifications (Optional)
- Embeds notes using SentenceTransformer
- Generates GPT-based explanations for H1 predictions
- Output: Elasticsearch index with justifications

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sql_download` | `True` | Download from SQL (False = load from ES) |
| `enable_nlp` | `True` | Run SpaCy NLP enrichment |
| `build_models` | `True` | Train models |
| `run_predictions` | `True` | Generate predictions |
| `build_rag` | `True` | Build RAG index and GPT justifications |
| `training_length` | `5` | Minimum notes required for training |
| `es_host` | `localhost:9200` | Elasticsearch endpoint |

## Elasticsearch Indices

| Index Name | Description |
|------------|-------------|
| `scorecard_details` | Raw SQL data backup |
| `scorecard_enriched` | NLP-enriched notes |
| `scorecard_sid_history` | Sliding window data |
| `scorecard_predictions` | H1 predictions |
| `scorecard_predictions_h2` | H2 predictions |
| `scorecard_model_summary` | Model performance metrics |
| `scorecard_rag_notes` | Embeddings + GPT justifications |
| `scorecard_log` | Pipeline execution logs |

## Programmatic Usage

### Train Models for Specific Horizon

```python
from scorecard import Horizon

# Train only H1 model
modeler.find_best_model(horizon=Horizon.H1)

# Train only H2 model
modeler.find_best_model(horizon=Horizon.H2)
```

### Generate Predictions for Specific Horizon

```python
# H1 predictions (backward compatible)
h1_preds = modeler.predict_with_best_model(state.sid_df, horizon=Horizon.H1)

# H2 predictions
h2_preds = modeler.predict_with_best_model(state.sid_df, horizon=Horizon.H2)
```

### Access Per-Horizon Results

```python
# Best model for H1
h1_model = state.best_model_by_horizon[1]
h1_key = state.best_model_key_by_horizon[1]

# Best model for H2
h2_model = state.best_model_by_horizon[2]
h2_key = state.best_model_key_by_horizon[2]

# Predictions DataFrames
h1_df = state.predictions_df_by_horizon[1]
h2_df = state.predictions_df_by_horizon[2]
```

### Load Model by Key

```python
# Legacy format (H1)
model_key = "complete_main_words_only | no_downsample_weighted | count | {0: 0.5, 1: 1.35, 2: 1.15}"
modeler.load_best_model_by_key(model_key, horizon=Horizon.H1)

# New format with horizon prefix (H2)
model_key = "H2 | complete_main_words_only | no_downsample_weighted | count | {0: 0.5, 1: 1.35, 2: 1.15}"
modeler.load_best_model_by_key(model_key)  # Horizon parsed from key
```

## Requirements

- Python 3.10+
- GPU recommended for SpaCy transformer model and embeddings
- Elasticsearch 8.x running locally or accessible
- SQL Server access (or pre-loaded ES data)

## License

Internal use only - Lockheed Martin Subcontract Scorecard System
