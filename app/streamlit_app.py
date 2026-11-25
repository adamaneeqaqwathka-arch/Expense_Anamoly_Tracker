"""Expense Anomaly Auditor: the Streamlit prototype.

Run the API first (uvicorn api.main:app), then:   streamlit run app/streamlit_app.py
Set ANOMALY_API_URL if the API is not at http://127.0.0.1:8000.
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `app` and `api` import when run by Streamlit

import pandas as pd
import streamlit as st

from app import helpers
from app.client import ApiError, AuditorClient

ROW_LIMIT = 1000
DEFAULT_URL = os.getenv("ANOMALY_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Expense Anomaly Auditor", page_icon="🔍", layout="wide")


@st.cache_resource
def get_client(url: str) -> AuditorClient:
    return AuditorClient(url)


@st.cache_data(show_spinner=False)
def fetch_report(_client: AuditorClient, url: str, run_id: str, rate: float, flagged_only: bool) -> bytes:
    return _client.report(run_id, rate, flagged_only)


def reset_results() -> None:
    for key in ("result", "detection", "upload_key"):
        st.session_state.pop(key, None)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("API address", DEFAULT_URL)
    client = get_client(api_url)
    try:
        info = client.health()
        st.success(f"API online (v{info['version']})")
        api_ok = True
    except ApiError as err:
        st.error(err.detail)
        api_ok = False
    st.divider()
    st.caption("A flag means a transaction is **unusual**, not that it is fraudulent. "
               "A person should review every flag.")

st.title("🔍 Expense Anomaly Auditor")
st.write("Upload your transactions. The app learns what is normal **in your file** and ranks every transaction "
         "by how unusual it is, with a reason for each flag.")

if not api_ok:
    st.info("Start the API with `uvicorn api.main:app`, then reload this page.")
    st.stop()

# ------------------------------------------------------------------ 1. upload
uploaded = st.file_uploader("1. Upload transactions (CSV)", type="csv", on_change=reset_results)
if uploaded is None:
    st.stop()

content = uploaded.getvalue()
upload_key = (uploaded.name, len(content))
if st.session_state.get("upload_key") != upload_key:
    try:
        with st.spinner("Reading the file..."):
            st.session_state["detection"] = client.detect_columns(uploaded.name, content)
        st.session_state["upload_key"] = upload_key
        st.session_state.pop("result", None)
    except ApiError as err:
        st.error(err.detail)
        st.stop()
detection = st.session_state["detection"]

with st.expander(f"File preview ({detection['total_rows']:,} rows)", expanded=False):
    st.dataframe(pd.DataFrame(detection["preview"]), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ 2. column mapping
st.subheader("2. Check the columns")
st.caption("The app guessed which column is which. Correct anything that is wrong.")
options = [helpers.NONE_CHOICE, *detection["columns"]]
chosen: dict[str, str] = {}
required = detection["required_roles"]
roles = [*required, *detection["optional_roles"]]
grid = st.columns(4)
for i, role in enumerate(roles):
    guess = detection["detected"].get(role)
    label = helpers.ROLE_LABELS.get(role, role) + (" *" if role in required else "")
    chosen[role] = grid[i % 4].selectbox(label, options, index=options.index(guess) if guess in options else 0,
                                         key=f"map_{role}_{upload_key}")
missing = [r for r in required if chosen[r] == helpers.NONE_CHOICE]
if missing:
    st.warning("Choose a column for: " + ", ".join(helpers.ROLE_LABELS[r] for r in missing))

# ------------------------------------------------------------------ 3. run
st.subheader("3. Run the audit")
if st.button("Find unusual transactions", type="primary", disabled=bool(missing)):
    overrides = helpers.overrides_from_choices(detection["detected"], chosen)
    try:
        with st.spinner("Learning what is normal and scoring every transaction..."):
            st.session_state["result"] = client.score(uploaded.name, content, flag_rate=0.05,
                                                     columns=overrides, limit=ROW_LIMIT)
    except ApiError as err:
        st.error(err.detail)

result = st.session_state.get("result")
if result is None:
    st.stop()

# ------------------------------------------------------------------ 4. results
st.divider()
st.subheader("4. Results")
percent = st.slider("Sensitivity: flag the most unusual …% of transactions", 1, 10, round(result["flag_rate"] * 100),
                    help="Higher flags more transactions. Changing it does not re-run the model.")
rate = percent / 100
if abs(rate - result["flag_rate"]) > 1e-9:
    try:
        result = client.get_run(result["run_id"], rate, ROW_LIMIT)
        st.session_state["result"] = result
    except ApiError as err:
        if err.status == 404:
            st.warning("This result has expired. Please run the audit again.")
            reset_results()
        else:
            st.error(err.detail)
        st.stop()

summary = result["summary"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Transactions used", f"{summary['transactions_used']:,}")
m2.metric("Flagged", f"{result['flagged_count']:,}")
m3.metric("Behaviour groups found", summary["clusters"])
m4.metric("Cluster separation", f"{summary['silhouette']:.2f}",
          help="Silhouette score: closer to 1 means the groups are clearly separated.")

tab_flags, tab_charts, tab_groups, tab_data = st.tabs(["Flagged transactions", "Charts", "Behaviour groups",
                                                        "Data cleaning"])

with tab_flags:
    table = helpers.flagged_table(result["rows"])
    if table.empty:
        st.info("Nothing flagged.")
    else:
        st.dataframe(table, hide_index=True, use_container_width=True,
                     column_config={"Amount": st.column_config.NumberColumn(format="%.2f"),
                                    "Why it was flagged": st.column_config.TextColumn(width="large")})
        if result["flagged_count"] > len(table):
            st.caption(f"Showing the {len(table):,} most unusual of {result['flagged_count']:,} flagged. "
                       "The download has all of them.")
    c1, c2 = st.columns(2)
    c1.download_button("Download flagged transactions (CSV)",
                       fetch_report(client, api_url, result["run_id"], rate, True),
                       file_name="flagged_transactions.csv", mime="text/csv")
    c2.download_button("Download full audit report (CSV)",
                       fetch_report(client, api_url, result["run_id"], rate, False),
                       file_name="audit_report.csv", mime="text/csv")

with tab_charts:
    charts = result["charts"]
    st.plotly_chart(helpers.score_histogram(charts, helpers.flag_threshold(result["rows"])),
                    use_container_width=True)
    left, right = st.columns(2)
    left.plotly_chart(helpers.flagged_by_category(charts), use_container_width=True)
    right.plotly_chart(helpers.flagged_over_time(charts), use_container_width=True)

with tab_groups:
    st.caption("Transactions with similar behaviour are grouped. A group holding very few transactions "
               "is a rare pattern and is treated as suspicious.")
    st.dataframe(helpers.cluster_table(summary), hide_index=True, use_container_width=True)
    st.write("Features used: " + ", ".join(f"`{f}`" for f in summary["features"]))

with tab_data:
    for line in helpers.cleaning_lines(summary):
        st.write("• " + line)
    st.write("Columns used: " + ", ".join(f"{helpers.ROLE_LABELS.get(r, r)} → `{c}`"
                                          for r, c in result["columns"].items()))

# ------------------------------------------------------------------ 5. single transaction
with st.expander("Check one new transaction against this file"):
    with st.form("single"):
        a, b = st.columns(2)
        amount = a.number_input("Amount", min_value=0.01, value=100.0, step=10.0)
        when = b.date_input("Date", value=date.today())
        c, d, e = st.columns(3)
        category = c.text_input("Category (optional)")
        merchant = d.text_input("Merchant (optional)")
        card = e.text_input("Card or customer ID (optional)")
        go = st.form_submit_button("Check it")
    if go:
        try:
            out = client.score_transaction(result["run_id"], amount, datetime.combine(when, datetime.min.time())
                                           .replace(hour=12).isoformat(), category, merchant, card)
            if out["would_be_flagged"]:
                st.error(f"Would be flagged. {out['reason']}")
            else:
                st.success("Would not be flagged.")
            st.caption(f"More unusual than {out['risk_percentile']:.0f}% of the transactions in this file "
                       f"(score {out['anomaly_score']:.2f}).")
        except ApiError as err:
            st.error(err.detail)