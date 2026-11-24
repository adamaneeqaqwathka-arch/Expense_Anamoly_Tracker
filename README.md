# Expense Anomaly Auditor

Flags unusual transactions with K-Means. One scoring engine, used by an API, a Streamlit prototype and a website.

```
Website ─┐
         ├─► FastAPI ─► anomaly_auditor (engine) ─► saved model files
Streamlit ┘
```

## Project layout

| Folder | What lives there | Build step |
|--------|------------------|-----------|
| `src/anomaly_auditor/` | The engine: schema, cleaning, features, scoring, explanations | 2 |
| `api/` | FastAPI backend | 3 |
| `app/` | Streamlit prototype | 4 |
| `web/` | Website | 5 |
| `tests/` | pytest tests | every step |
| `notebooks/` | The analysis notebook | done |
| `data/`, `models/`, `outputs/` | Local data, saved models, reports (not committed) | |
| `docs/` | Design decisions | 0 |

## Setup

```bash
# 1. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. install the project and the dev tools
pip install -r requirements-dev.txt

# 3. check everything works
pytest
```

All tests should pass. If `import anomaly_auditor` fails, step 2 did not finish.

## Roadmap

See the build roadmap: engine, API, Streamlit, website, validation, deployment.

## Status

Step 1 (project setup) done. Next: Step 2, the engine.
