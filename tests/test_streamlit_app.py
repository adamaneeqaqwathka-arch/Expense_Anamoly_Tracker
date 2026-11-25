"""The Streamlit screen starts cleanly, also when the API is not running."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_starts_and_explains_when_the_api_is_down(monkeypatch):
    monkeypatch.setattr("os.environ", {**__import__("os").environ, "ANOMALY_API_URL": "http://127.0.0.1:9"})
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value.endswith("Expense Anomaly Auditor")
    assert any("Cannot reach the API" in e.value for e in at.error)
    assert any("uvicorn api.main:app" in i.value for i in at.info)