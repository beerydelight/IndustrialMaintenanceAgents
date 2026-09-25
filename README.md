# Industrial Maintenance Agents

An AI-assisted industrial maintenance workflow for diagnosing equipment alerts,
checking historical maintenance records, and estimating failure risk with a
local digital-twin model.

The application is designed to run locally. Ollama provides the language-model
calls, while the digital-twin tool uses a deterministic machine-learning and
rule-based pipeline. No cloud LLM service is required.

## What it does

Given a natural-language equipment alert, the supervisor:

1. Extracts and normalizes an equipment identifier.
2. Runs the real-time maintenance agent, which extracts sensor values, checks
   replacement-part inventory, retrieves relevant maintenance documents, and
   asks Ollama for a schema-validated diagnosis.
3. Runs the historical archive agent against the local SQLite maintenance log.
4. Runs the deterministic digital-twin tool to train or refresh a model,
   predict failure probability, and apply maintenance alert rules.
5. Combines the worker results into an executive maintenance report and records
   the completed result in the maintenance database.

If Ollama is unavailable or returns invalid output, the workers use their
explicit fallback paths rather than returning an unstructured result.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) running locally
- The Ollama model configured by `OLLAMA_MODEL` (defaults to
  `llama3.2:latest`)
- Enough local disk and memory for PyTorch, ChromaDB, and the
  `all-MiniLM-L6-v2` sentence-transformer embedding model

## Installation

```bash
git clone https://github.com/beerydelight/IndustrialMaintenanceAgents.git
cd IndustrialMaintenanceAgents

python -m venv .venv
. .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Pull the default Ollama model before running the workflow:

```bash
ollama pull llama3.2:latest
```

The RAG pipeline loads its embedding model locally. The application sets
Hugging Face offline/no-progress environment flags in `main.py`, so the
embedding model must already be available in the local Hugging Face cache.

## Configuration

The following environment variables are optional:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2:latest` | Ollama model tag |
| `MAINTENANCE_LOG_DB` | `data/maintenance.db` | SQLite maintenance-log path |
| `KAGGLE_USERNAME` | unset | Optional Kaggle account for dataset ingestion |
| `KAGGLE_KEY` | unset | Optional Kaggle API key for dataset ingestion |

For example:

```bash
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.2:latest
```

Do not commit credentials. The optional Kaggle path is only used by the
non-deterministic full digital-twin pipeline; the supervisor’s
`digital_twin_tool` uses deterministic synthetic data and does not require
Kaggle credentials.

## Run the CLI

Start Ollama, activate the virtual environment, and provide an alert either as
an argument or interactively:

```bash
python main.py "Pump PU-118-07 pressure fell to 0.4 bar and vibration reached 9.2 mm/s"
```

```bash
python main.py
# Describe the equipment issue: ...
```

The command writes operational logs to stderr and the final executive report
to stdout. Equipment identifiers are recognized in forms such as
`PU-118-07`, `PU_118_07`, or `pump number 7`.

## Seed or refresh maintenance history

The repository includes a sample SQLite database at
`data/maintenance.db`. Recreate it with the sample records when needed:

```bash
python utils/db_seeder.py
```

This command drops and recreates the `maintenance_logs` table, so it should not
be used against a database containing records that must be preserved.

## Digital-twin interfaces

The `digital_twin` package exposes three Python entry points:

```python
from digital_twin import digital_twin_tool, predict, run_twin_pipeline

# Supervisor-safe deterministic analysis from an alert or readings dictionary.
result = digital_twin_tool(
    "Motor temperature reached 85 C, pressure was 17 bar, vibration was 9 mm/s"
)

# Full training pipeline; may use Kaggle data when credentials are configured.
trained = run_twin_pipeline("Motor temperature reached 85 C")

# Inference-only prediction after a model has been trained.
prediction = predict({"temperature_c": 85.0, "vibration_mm_s": 9.0})
```

The deterministic tool extracts supported readings from alert text, trains a
model using generated sensor data, imputes missing features, and returns a
failure probability, status, supplied/imputed features, and rule-based alerts.
Generated artifacts are stored under `~/digital_twin_data/outputs/`, including
sensor data, model weights, operating parameters, training statistics, and
alert history.

`digital_twin/api.py` contains FastAPI router handlers for prediction, sensor
history, alerts, model metadata, and operating parameters. It is a router
module, not a standalone ASGI application; mount `digital_twin.api.router` in
your own FastAPI app if HTTP endpoints are required.

## Repository layout

```text
.
├── agents/
│   ├── supervisor_agent.py   # LangGraph orchestration and report synthesis
│   ├── maintenance_agent.py  # Current-alert diagnosis
│   └── archiver_agent.py     # Historical-log analysis
├── core/
│   ├── rag_pipeline.py       # ChromaDB and sentence-transformers retrieval
│   ├── tools.py              # Alert parsing, inventory, model, and fallbacks
│   ├── db_logic.py           # SQLite maintenance-log access
│   └── external_tools.py     # Archive-agent database adapters
├── digital_twin/
│   ├── pipeline.py           # Training and feature-engineering pipeline
│   ├── tool.py               # Deterministic supervisor tool
│   ├── api.py                # FastAPI router
│   ├── model/                # Dynamic model training and inference
│   └── data_ingestion/       # Synthetic and optional Kaggle ingestion
├── data/
│   ├── maintenance.db        # Sample historical maintenance records
│   └── ai4i2020.csv          # Predictive-maintenance source dataset
├── utils/
│   ├── db_seeder.py          # Rebuild the sample SQLite database
│   └── dataset_downloader.py # Optional Kaggle dataset downloader
├── main.py                   # CLI entry point
├── agent.py                  # Backward-compatible maintenance-agent import
└── requirements.txt
```

The persistent `chroma_db/` directory is used by the RAG pipeline. Generated
digital-twin files are intentionally kept outside the repository under
`~/digital_twin_data/`.

## Diagnostic output

The maintenance worker validates its internal response against this schema:

```json
{
  "status": "OK|WARNING|CRITICAL",
  "diagnosis": "Concise diagnosis",
  "action_recommended": "Specific technician action",
  "inventory_status": "available|low|critical"
}
```

The user-facing output is an executive report that also includes equipment
identity, digital-twin status and failure probability, historical risk, and
maintenance history. It is intentionally human-readable rather than raw JSON.

## Optional Kaggle ingestion

To download configured datasets for the full training pipeline:

```bash
python utils/dataset_downloader.py
```

Set `KAGGLE_USERNAME` and `KAGGLE_KEY` first. The downloader caches files in
the project’s `data/` directory and skips datasets that are already present.
