"""Smoke tests for every dashboard page using Streamlit's built-in testing framework (AppTest).

Each page is rendered headlessly; the test fails if any page raises an exception. The
prediction page is also driven through its form (valid input, missing bills, and the case
where the most recent bill is empty).
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
needs_artifacts = pytest.mark.skipif(not (ROOT / "artifacts" / "model_metadata.json").exists()
                                     or not (ROOT / "models" / "best_model.joblib").exists(),
                                     reason="saved model / artifacts not available")
st_testing = pytest.importorskip("streamlit.testing.v1")


def page_app(module: str):
    script = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import streamlit as st
st.set_page_config(layout="wide")
from dashboard.common import style
from dashboard import {module}
style()
{module}.render()
"""
    return st_testing.AppTest.from_string(script, default_timeout=180)


@needs_artifacts
@pytest.mark.parametrize("module", ["page_overview", "page_explore", "page_compare", "page_performance",
                                    "page_predict"])
def test_page_renders_without_error(module):
    at = page_app(module).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.title) + len(at.markdown) > 0


@needs_artifacts
def test_prediction_form_valid_and_missing_inputs():
    at = page_app("page_predict").run()
    assert not at.exception
    at.button[1].click().run()                       # "Household with air conditioning" example
    assert at.number_input(key="bill_0").value == 265.0
    at.button[3].click().run()                       # example with missing bills
    assert at.number_input(key="bill_1").value is None
    submit = [b for b in at.button if "Predict" in str(b.label)]
    assert submit, "submit button not found"
    submit[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "Estimated Monthly Electricity Consumption" in html and "kWh" in html

    # most recent bill left empty -> friendly error, no crash
    at.number_input(key="bill_0").set_value(None)
    [b for b in at.button if "Predict" in str(b.label)][0].click().run()
    assert not at.exception
    assert any("most recent bill" in e.value for e in at.error)

    # a cleared bill in the middle must stay MISSING after submitting (regression test: Streamlit would
    # otherwise fall back to min_value=0 and silently turn a missing bill into a 0 kWh bill)
    at.number_input(key="bill_0").set_value(120.0)
    at.number_input(key="bill_4").set_value(None)
    [b for b in at.button if "Predict" in str(b.label)][0].click().run()
    assert at.number_input(key="bill_4").value is None
    assert any("Estimated Monthly Electricity Consumption" in m.value for m in at.markdown)


@needs_artifacts
def test_full_app_entrypoint_runs():
    at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
