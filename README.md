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
| `app/` | Streamlit prototype and the API client | 4 |
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

## Use the engine

```python
import pandas as pd
from anomaly_auditor.engine import AnomalyEngine

raw = pd.read_csv("data/my_transactions.csv")
engine = AnomalyEngine()                       # flag_rate=0.05 flags the top 5%
ranked = engine.fit_score(raw)                 # detect columns, clean, learn normal, score
print(ranked[["rank", "amount", "category", "anomaly_score", "flagged", "reason"]].head(10))

engine.flag(ranked, flag_rate=0.02)            # change sensitivity without re-scoring (max 0.10)
engine.summary()                               # clusters, silhouette, features, cleaning report
engine.save("models/engine.joblib")            # AnomalyEngine.load(...) restores it
```

If a column is not detected, name it: `engine.fit_score(raw, overrides={"amount": "Total Spend"})`.

## Run the API

```bash
uvicorn api.main:app --reload        # then open http://127.0.0.1:8000/docs
```

| Endpoint | What it does |
|----------|--------------|
| `GET /health` | Is the service up |
| `GET /model/info` | Method, defaults, limits, recognised column names |
| `POST /columns/detect` | Upload a CSV, see which column the engine would use for each role |
| `POST /score` | Upload a CSV (+ `flag_rate`, optional `columns` JSON, `limit`); get the ranked table, summary and chart data |
| `GET /runs/{run_id}` | Same result with a different `flag_rate` (the sensitivity slider), no re-upload |
| `GET /runs/{run_id}/report.csv` | Download the audit report |
| `POST /score/transaction` | Score one new transaction against a scored file |

```bash
curl -F "file=@data/my_transactions.csv" -F "flag_rate=0.03" http://127.0.0.1:8000/score
```

Limits and CORS can be changed with environment variables: `ANOMALY_MAX_UPLOAD_MB`, `ANOMALY_MAX_ROWS`,
`ANOMALY_MAX_RUNS`, `ANOMALY_RUN_TTL_SECONDS`, `ANOMALY_MAX_TABLE_ROWS`, `ANOMALY_CORS_ORIGINS` (comma separated).
Scored files are kept in memory for one hour, so a restart clears them.

## Run the Streamlit prototype

Start the API, then in a second terminal:

```bash
streamlit run app/streamlit_app.py          # http://localhost:8501
```

If the API is not at `http://127.0.0.1:8000`, set `ANOMALY_API_URL` (or edit the address in the sidebar).
The app uploads a CSV, shows the detected columns for you to correct, runs the audit through the API, and
offers a sensitivity slider (no re-run), charts, behaviour groups, CSV downloads and a one-transaction checker.
It holds no scoring logic: everything goes through `app/client.py`.

## How the score works

1. Features are built per transaction (amount compared with its category, merchant rarity, time of day, gap since the card's previous transaction, category flags) and standardised.
2. K-Means groups transactions; k is chosen automatically by silhouette score.
3. **Behaviour signal:** distance to the nearest *normal* cluster centre, normalised inside that cluster. Clusters holding under 2% of transactions are not treated as normal behaviour and add extra weight.
4. **Amount signal:** how far the amount is from what is typical for its category.
5. Both signals are added, rows are ranked, and the top `flag_rate` are flagged with a plain-language reason.

A flag means *unusual*, not *fraudulent*. A person decides.

## Roadmap

See the build roadmap: engine, API, Streamlit, website, validation, deployment.

## Status

Steps 1 (setup), 2 (engine), 3 (API) and 4 (Streamlit prototype) done; 98 tests pass. Next: Step 5, the website.