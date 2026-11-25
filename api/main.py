"""Step 3: the FastAPI backend. Routes are thin; all analysis happens in the anomaly_auditor engine.

Run it:   uvicorn api.main:app --reload       then open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from anomaly_auditor import __version__
from anomaly_auditor.cleaning import clean
from anomaly_auditor.engine import AnomalyEngine, EngineError
from anomaly_auditor.schema import ALIASES, REQUIRED, ROLES, ColumnMap, SchemaError, detect_columns

from .schemas import (
    ColumnDetection,
    ErrorOut,
    Health,
    ModelInfo,
    ScoreResponse,
    TransactionIn,
    TransactionOut,
)
from .settings import Settings
from .store import Run, RunStore

TABLE_COLUMNS = ["rank", "source_row", "date", "amount", "category", "merchant", "card", "cluster",
                 "anomaly_score", "risk_percentile", "flagged", "reason"]
ERRORS = {400: {"model": ErrorOut}, 404: {"model": ErrorOut}, 413: {"model": ErrorOut}, 422: {"model": ErrorOut}}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = RunStore(settings.max_runs, settings.run_ttl_seconds)

    app = FastAPI(title="Expense Anomaly Auditor", version=__version__,
                  description="Upload transaction data, get back a ranked list of unusual transactions. "
                              "A flag means unusual, not fraudulent: a person decides.")
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=["*"], allow_headers=["*"])
    app.state.settings, app.state.store = settings, store

    # ---------------------------------------------------------------- helpers
    def read_csv(upload: UploadFile) -> pd.DataFrame:
        name = (upload.filename or "").lower()
        if not name.endswith(".csv"):
            raise HTTPException(400, "Please upload a .csv file.")
        limit = settings.max_upload_mb * 1024 * 1024
        data = upload.file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"The file is larger than {settings.max_upload_mb} MB.")
        try:
            df = pd.read_csv(io.BytesIO(data))
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as err:
            raise HTTPException(400, f"The file could not be read as a CSV: {err}") from err
        if len(df) > settings.max_rows:
            raise HTTPException(413, f"The file has more than {settings.max_rows:,} rows.")
        return df

    def parse_overrides(text: str | None) -> dict[str, str | None] | None:
        if not text:
            return None
        try:
            value = json.loads(text)
        except json.JSONDecodeError as err:
            raise HTTPException(400, f"'columns' must be a JSON object such as {{\"amount\": \"Total\"}}: {err}") from err
        if not isinstance(value, dict):
            raise HTTPException(400, "'columns' must be a JSON object such as {\"amount\": \"Total\"}.")
        return value

    def get_run(run_id: str) -> Run:
        run = store.get(run_id)
        if run is None:
            raise HTTPException(404, "Unknown or expired run_id. Upload the file again with POST /score.")
        return run

    def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        out = frame.reset_index()
        cols = [c for c in TABLE_COLUMNS if c in out.columns]
        return json.loads(out[cols].to_json(orient="records", date_format="iso"))

    def charts(table: pd.DataFrame) -> dict[str, Any]:
        scores = table["anomaly_score"].to_numpy()
        top = float(np.quantile(scores, 0.995))
        edges = np.linspace(float(scores.min()), max(top, float(scores.min()) + 1e-9), 31)
        counts, _ = np.histogram(np.clip(scores, edges[0], edges[-1]), bins=edges)
        by_cat = (table.groupby("category")["flagged"].agg(flagged="sum", total="size")
                  .sort_values(["flagged", "total"], ascending=False).head(10))
        by_month = table.groupby(table["date"].dt.to_period("M").astype(str))["flagged"].agg(flagged="sum", total="size")
        return {
            "score_histogram": {"bin_edges": edges.round(4).tolist(), "counts": counts.tolist(),
                                "clipped_above": round(top, 4)},
            "flagged_by_category": by_cat.reset_index().to_dict(orient="records"),
            "flagged_by_month": by_month.reset_index().rename(columns={"date": "month"}).to_dict(orient="records"),
        }

    def payload(run: Run, flag_rate: float, limit: int) -> dict[str, Any]:
        try:
            table = run.engine.flag(run.table, flag_rate)
        except EngineError as err:
            raise HTTPException(422, str(err)) from err
        limit = min(limit, settings.max_table_rows)
        return {
            "run_id": run.run_id, "filename": run.filename, "flag_rate": flag_rate,
            "total_rows": len(table), "flagged_count": int(table["flagged"].sum()),
            "columns": run.engine.columns_.used(), "summary": run.engine.summary(),
            "rows": records(table.head(limit)), "rows_returned": min(limit, len(table)),
            "charts": charts(table),
        }

    # ---------------------------------------------------------------- routes
    @app.get("/health", response_model=Health, tags=["status"])
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/model/info", response_model=ModelInfo, tags=["status"])
    def model_info():
        """How the engine works, its defaults and limits, and which column names it recognises."""
        probe = AnomalyEngine()
        return {
            "version": __version__,
            "method": "K-Means behaviour clusters plus a category-relative amount signal, fitted on the uploaded file",
            "defaults": {"flag_rate": probe.flag_rate, "max_flag_rate": probe.MAX_FLAG_RATE,
                         "min_rows": probe.min_rows, "k_range": list(probe.k_range),
                         "small_cluster_cut": probe.small_cluster_cut},
            "limits": {"max_upload_mb": settings.max_upload_mb, "max_rows": settings.max_rows,
                       "max_table_rows": settings.max_table_rows, "run_ttl_seconds": settings.run_ttl_seconds},
            "roles": {"required": list(REQUIRED), "optional": [r for r in ROLES if r not in REQUIRED],
                      **{f"aliases_{role}": list(names) for role, names in ALIASES.items()}},
        }

    @app.post("/columns/detect", response_model=ColumnDetection, responses=ERRORS, tags=["scoring"])
    def detect(file: UploadFile = File(..., description="the transactions as a CSV file")):
        """Look at a CSV and say which column the engine would use for each role, so a user can correct it."""
        df = read_csv(file)
        return {"columns": [str(c) for c in df.columns], "detected": detect_columns(df),
                "required_roles": list(REQUIRED), "optional_roles": [r for r in ROLES if r not in REQUIRED],
                "preview": json.loads(df.head(5).to_json(orient="records", date_format="iso")),
                "total_rows": len(df)}

    @app.post("/score", response_model=ScoreResponse, responses=ERRORS, tags=["scoring"])
    def score(
        file: UploadFile = File(..., description="the transactions as a CSV file"),
        flag_rate: float = Form(0.05, gt=0, le=0.10, description="share of transactions to flag (sensitivity)"),
        columns: str | None = Form(None, description='optional JSON mapping, e.g. {"amount": "Total Spend"}'),
        limit: int = Form(200, ge=1, description="how many ranked rows to return"),
    ):
        """Learn what is normal in the uploaded file, then rank every transaction by how unusual it is."""
        df = read_csv(file)
        engine = AnomalyEngine(flag_rate=flag_rate)
        try:
            table = engine.fit_score(df, overrides=parse_overrides(columns))
        except (SchemaError, EngineError) as err:
            raise HTTPException(422, str(err)) from err
        run = store.add(Run(engine=engine, table=table, filename=file.filename or "upload.csv"))
        return payload(run, flag_rate, limit)

    @app.get("/runs/{run_id}", response_model=ScoreResponse, responses=ERRORS, tags=["runs"])
    def get_result(run_id: str,
                   flag_rate: float = Query(0.05, gt=0, le=0.10),
                   limit: int = Query(200, ge=1)):
        """Re-read a scored file with a different sensitivity. No new upload and no re-scoring."""
        return payload(get_run(run_id), flag_rate, limit)

    @app.get("/runs/{run_id}/report.csv", responses=ERRORS, tags=["runs"])
    def report(run_id: str, flag_rate: float = Query(0.05, gt=0, le=0.10), flagged_only: bool = False):
        """Download the audit report as a CSV: every transaction ranked, with score, flag and reason."""
        run = get_run(run_id)
        table = run.engine.flag(run.table, flag_rate)
        if flagged_only:
            table = table[table["flagged"]]
        body = table.reset_index().to_csv(index=False)
        return Response(body, media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="audit_report_{run_id}.csv"'})

    @app.post("/score/transaction", response_model=TransactionOut, responses=ERRORS, tags=["scoring"])
    def score_transaction(item: TransactionIn):
        """Score one new transaction against what is normal in an already scored file."""
        run = get_run(item.run_id)
        fields = {"amount": item.amount, "date": item.date, "category": item.category,
                  "merchant": item.merchant, "card": item.card}
        fields = {k: v for k, v in fields.items() if v is not None}
        try:
            cmap = ColumnMap(amount="amount", date="date", category=fields.get("category") and "category",
                             merchant=fields.get("merchant") and "merchant", card=fields.get("card") and "card")
            cleaned, _ = clean(pd.DataFrame([fields]), cmap)
            row = run.engine.score(cleaned).iloc[0]
        except (SchemaError, EngineError) as err:
            raise HTTPException(422, str(err)) from err
        scores = run.table["anomaly_score"].to_numpy()
        flagged = run.engine.flag(run.table)
        threshold = float(flagged.loc[flagged["flagged"], "anomaly_score"].min())
        return {"anomaly_score": float(row["anomaly_score"]),
                "risk_percentile": float((scores <= row["anomaly_score"]).mean() * 100),
                "would_be_flagged": bool(row["anomaly_score"] >= threshold),
                "reason": str(row["reason"]), "cluster": int(row["cluster"])}

    return app


app = create_app()
