# Industrial Maintenance AI Agent

This workspace contains a production-oriented maintenance diagnostics agent that reads simulated machinery alert text and uses Ollama with Llama 3.2 to produce a JSON diagnostic response.

## Features

- ReAct-style reasoning loop with a retry-safe failure guard
- Strict JSON schema enforcement for diagnostics output
- Deterministic sensory-data extraction and replacement-part inventory checks
- Deterministic digital-twin ML/rules pipeline invoked by the supervisor as a tool
- Local Ollama integration using `llama3.2:latest`
- Simulated industrial equipment alert ingestion from `data.json`
- Graceful fallback for service or parsing errors

## Files

- `agents/maintenance_agent.py` — current-alert diagnostics agent
- `agents/archiver_agent.py` — historical maintenance-log analysis agent
- `core/rag_pipeline.py` — shared retrieval-augmented generation pipeline
- `core/db_logic.py` and `core/external_tools.py` — shared SQLite history access
- `db_seeder.py` — creates `data/maintenance.db` with sample maintenance history
- `main.py` — CLI entry point
- `tools.py` — maintenance input parsing, sensor extraction, inventory checks, and model helpers
- `data/data.json` — simulated machinery sensory alert data
- `requirements.txt` — Python dependencies for the environment
- `.env` — local Ollama configuration

## Run

```bash
cd /home/amine_pc/apps/formation
. .venv/bin/activate
python main.py
```

To create or refresh the historical maintenance database:

```bash
python db_seeder.py
```

Select `1` to run the maintenance agent with `data/data.json`, or `2` to enter
an equipment ID for historical analysis. If the pipeline file has not yet been
moved in a fresh checkout, run `mv rag_pipeline.py core/rag_pipeline.py` from
the project root (the current layout already contains it in `core/`).

Option `1` runs the coordinated workflow: it extracts the equipment ID from the
alert, runs `MaintenanceAgent`, queries the same equipment's recent history with
`ArchiverAgent`, and compares matching issue terms against recent maintenance
records. The result includes a `historical_comparison.recommendation` value such
as `verify the previous repair`, `recurring issue requiring escalation`, or
`no matching historical intervention found`.

## Expected output schema

```json
{
  "status": "CRITICAL|WARNING|OK",
  "diagnosis": "summary of machine health",
  "action_recommended": "specific technician action",
  "inventory_status": "available|low|critical"
}
```

## Notes

The agent runs `get_sensory_data` and `check_inventory` before calling the model, so diagnostics include structured sensor readings and replacement-part availability. The retry loop is capped at `MAX_RETRIES = 5` to prevent closed-loop failures.

The supervisor also invokes `digital_twin_tool` for a deterministic train/infer/rules
analysis. That tool uses seeded synthetic data, numeric sensor extraction, model
inference, and threshold-based alerts; it does not call an LLM or the external
Kaggle ingestion path.
