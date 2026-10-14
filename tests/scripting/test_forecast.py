# WriterAgent - AI Writing Assistant for LibreOffice
# Copyright (c) 2026 KeithCu (modifications and relicensing)
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for trusted forecast helpers."""

from __future__ import annotations

import json
from importlib.util import find_spec
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from plugin.scripting.forecast import anomaly_detection_time_series, decompose_time_series, forecast_time_series, run_forecast
from plugin.scripting.venv import forecast as venv_forecast


def _require_statsmodels_installed() -> None:
    if find_spec("statsmodels") is None:
        pytest.skip("statsmodels not installed")


def _seasonal_series(n: int = 36) -> pd.DataFrame:
    rows = []
    for i in range(n):
        month = (i % 12) + 1
        year = 2020 + i // 12
        seasonal = 10.0 * (1 if month <= 6 else -1)
        rows.append({"Date": f"{year}-{month:02d}-01", "Value": 100.0 + i + seasonal})
    return pd.DataFrame(rows)


def test_forecast_time_series_ok():
    data = _seasonal_series()
    result = forecast_time_series(data, periods=6, model="auto")
    assert result["status"] == "ok"
    assert result["helper"] == "forecast_time_series"
    assert "metrics" in result
    assert "tables" in result
    assert result["tables"][0]["name"] == "forecast"
    assert len(result["tables"][0]["rows"]) == 6


def test_decompose_time_series_ok():
    _require_statsmodels_installed()
    data = _seasonal_series()
    result = decompose_time_series(data, period=12)
    assert result["status"] == "ok"
    assert result["helper"] == "decompose_time_series"
    assert result["tables"][0]["name"] == "decomposition"
    assert "trend" in result["tables"][0]["columns"]


def test_run_forecast_dispatch():
    data = _seasonal_series()
    result = run_forecast({"helper": "forecast_time_series", "params": {"periods": 3}}, data)
    assert result["status"] == "ok"
    assert result["helper"] == "forecast_time_series"


def test_forecast_insufficient_data():
    data = pd.DataFrame({"Date": ["2024-01-01"], "Value": [1.0]})
    result = forecast_time_series(data)
    assert result["status"] == "error"
    assert result["code"] == "INSUFFICIENT_DATA"


def test_forecast_unknown_column():
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6, 7, 8]})
    result = forecast_time_series(data, date_col="Date", value_col="Value")
    assert result["status"] == "error"
    assert result["code"] == "UNKNOWN_COLUMN"


def test_decompose_missing_statsmodels():
    data = _seasonal_series()
    with patch("plugin.scripting.venv.forecast._require_statsmodels", return_value=None):
        result = decompose_time_series(data, period=12)
    assert result["status"] == "error"
    assert result["code"] == "MISSING_PACKAGE"


def test_forecast_moving_average_fallback():
    data = _seasonal_series()
    with patch("plugin.scripting.venv.forecast._require_statsmodels", return_value=None):
        result = forecast_time_series(data, periods=4, model="auto")
    assert result["status"] == "ok"
    assert result["metrics"]["model"] == "moving_average"


def test_run_forecast_unknown_helper():
    result = run_forecast({"helper": "not_a_helper"}, _seasonal_series())
    assert result["status"] == "error"
    assert result["code"] == "UNKNOWN_HELPER"


def _seasonal_series_with_spike(n: int = 36, spike_idx: int = 18, spike_amount: float = 500.0) -> pd.DataFrame:
    df = _seasonal_series(n)
    df.loc[spike_idx, "Value"] = float(df.loc[spike_idx, "Value"]) + spike_amount
    return df


def test_anomaly_detection_time_series_flags_spike():
    _require_statsmodels_installed()
    data = _seasonal_series_with_spike()
    result = anomaly_detection_time_series(data, period=12, threshold=3.0)
    assert result["status"] == "ok"
    assert result["helper"] == "anomaly_detection_time_series"
    assert result["metrics"]["n_anomalies"] >= 1
    table = result["tables"][0]
    assert table["name"] == "anomalies"
    assert table["columns"] == ["date", "observed", "expected", "residual", "score"]
    assert len(table["rows"]) >= 1


def test_anomaly_detection_missing_statsmodels():
    data = _seasonal_series_with_spike()
    with patch("plugin.scripting.venv.forecast._require_statsmodels", return_value=None):
        result = anomaly_detection_time_series(data, period=12)
    assert result["status"] == "error"
    assert result["code"] == "MISSING_PACKAGE"


def test_anomaly_detection_insufficient_data():
    _require_statsmodels_installed()
    data = _seasonal_series(n=12)
    result = anomaly_detection_time_series(data, period=12)
    assert result["status"] == "error"
    assert result["code"] == "INSUFFICIENT_DATA"


def test_run_forecast_anomaly_dispatch():
    _require_statsmodels_installed()
    data = _seasonal_series_with_spike()
    result = run_forecast({"helper": "anomaly_detection_time_series", "params": {"period": 12}}, data)
    assert result["status"] == "ok"
    assert result["helper"] == "anomaly_detection_time_series"


@pytest.mark.parametrize("model", ["moving_average"])
def test_forecast_time_series_models(model: str):
    data = _seasonal_series()
    result = forecast_time_series(data, periods=3, model=model)
    assert result["status"] == "ok"
    assert result["metrics"]["model"] == model


def _indexed(frame: pd.DataFrame) -> pd.Series:
    work = frame.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    return work.set_index("Date")["Value"]


def test_daily_series_infers_weekly_season_not_monthly():
    # 30 daily rows used to become period 12 because n >= 24.
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    frame = pd.DataFrame({"Date": dates, "Value": [100.0 + (i % 7) for i in range(30)]})
    assert venv_forecast._infer_seasonal_periods(_indexed(frame), None) == 7
    assert venv_forecast._infer_seasonal_periods(_indexed(frame), 12) == 12
    result = forecast_time_series(frame, periods=2, model="auto")
    assert result["status"] == "ok"
    assert result["metrics"].get("seasonal_periods", 7) == 7


def test_weekly_series_stays_trend_only():
    # 30 weeks is not two years, so a seasonal model must not run.
    dates = pd.date_range("2024-01-07", periods=30, freq="W")
    frame = pd.DataFrame({"Date": dates, "Value": [float(i) for i in range(30)]})
    assert venv_forecast._infer_seasonal_periods(_indexed(frame), None) is None
    result = forecast_time_series(frame, periods=3, model="auto")
    assert result["status"] == "ok"
    assert result["metrics"]["model"] != "holt_winters"
    assert result["metrics"].get("seasonal_periods") != 12
    assert any("not enough cycles" in flag for flag in result["flags"])


def test_monthly_series_infers_annual_season():
    frame = _seasonal_series(36)
    assert venv_forecast._infer_seasonal_periods(_indexed(frame), None) == 12


def test_weekday_forecast_skips_weekends():
    dates = pd.bdate_range("2024-01-02", periods=60)
    frame = pd.DataFrame({"Date": dates, "Value": [float(i) for i in range(60)]})
    assert venv_forecast._infer_seasonal_periods(_indexed(frame), None) == 5
    result = forecast_time_series(frame, periods=5, model="moving_average")
    assert result["status"] == "ok"
    assert result["metrics"]["freq"] == "B"
    assert not any("irregular" in flag.lower() for flag in result["flags"])
    for row in result["tables"][0]["rows"]:
        assert pd.Timestamp(row[0]).dayofweek < 5


def test_forecast_periods_are_capped():
    rows = [{"Date": f"2024-01-{day:02d}", "Value": float(day)} for day in range(1, 10)]
    result = forecast_time_series(pd.DataFrame(rows), periods=10**9, model="moving_average")
    assert result["status"] == "ok", result
    assert result["metrics"]["periods"] == 10_000
    assert len(result["tables"][0]["rows"]) == 10_000
    assert result["tables"][0]["truncated"] is False
    assert "periods capped at 10000" in result["flags"]


def test_holt_winters_intervals_use_simulate_not_get_prediction():
    _require_statsmodels_installed()
    result = forecast_time_series(
        _seasonal_series(),
        periods=4,
        model="holt_winters",
        seasonal_periods=12,
    )
    assert result["status"] == "ok", result
    table = result["tables"][0]
    assert table["columns"] == ["date", "forecast", "lower", "upper"]
    assert "confidence intervals unavailable" not in result["flags"]
    assert "interval_note" not in result["metrics"]
    for row in table["rows"]:
        point, lower, upper = row[1], row[2], row[3]
        assert lower <= point <= upper


def test_holt_winters_interval_failure_is_a_structured_note(monkeypatch):
    _require_statsmodels_installed()
    monkeypatch.setattr(venv_forecast, "_holt_winters_intervals", lambda fit, periods: None)
    result = forecast_time_series(
        _seasonal_series(),
        periods=3,
        model="holt_winters",
        seasonal_periods=12,
    )
    assert result["status"] == "ok", result
    assert "lower" not in result["tables"][0]["columns"]
    note = result["metrics"]["interval_note"]
    assert note["available"] is False
    assert note["method"] == "HoltWintersResults.simulate"
    assert "get_prediction" in note["message"]
    assert "confidence intervals unavailable" not in result["flags"]
    assert note["message"] in result["flags"]


def test_duplicate_dates_are_aggregated():
    rows = [{"Date": f"2024-01-{day:02d}", "Value": float(day)} for day in range(1, 10)]
    rows.append({"Date": "2024-01-01", "Value": 100.0})
    result = forecast_time_series(pd.DataFrame(rows), periods=2, model="moving_average")
    assert result["status"] == "ok"
    assert any("Aggregated 1 duplicate" in flag for flag in result["flags"])


def test_insert_forecast_result_into_calc() -> None:
    from unittest.mock import MagicMock, patch
    from plugin.scripting.forecast import insert_forecast_result_into_calc

    doc = MagicMock()
    ctx = MagicMock()
    result = {"status": "ok", "tables": [{"columns": ["Date", "Forecast"], "rows": [["2024-01-01", 10.0]]}]}

    with patch("plugin.calc.tabular_egress.insert_tabular_result_into_calc", return_value=1) as mock_insert:
        res = insert_forecast_result_into_calc(doc, ctx, result)
        assert res == 1
        mock_insert.assert_called_once()


def test_calc_serial_dates_are_not_1970():
    _require_statsmodels_installed()
    serials = [45292.0 + i for i in range(60)]
    values = [10.0 + float(i % 7) * 2.0 + float(i) * 0.1 for i in range(60)]
    frame = pd.DataFrame({"Date": serials, "Value": values})
    result = forecast_time_series(frame, periods=7, model="auto")
    assert result["status"] == "ok", result
    first_forecast_date = str(result["tables"][0]["rows"][0][0])
    assert first_forecast_date.startswith("2024-03-01")
    assert result["metrics"]["seasonal_periods"] == 7
    freq = str(result["metrics"].get("freq", ""))
    assert "ns" not in freq.lower()
    flags = result.get("flags", [])
    assert not any("season" in f.lower() and "could not be inferred" in f.lower() for f in flags)


def test_year_integers_forecast_the_next_year():
    frame = pd.DataFrame({"Year": list(range(2000, 2024)), "Value": [float(i) for i in range(24)]})
    result = forecast_time_series(frame, date_col="Year", periods=3)
    assert result["status"] == "ok", result
    rows = result["tables"][0]["rows"]
    assert str(rows[0][0]).startswith("2024-01-01")
    assert not any("1970" in str(r[0]) for r in rows)


def test_gapped_month_end_is_not_a_flat_average():
    _require_statsmodels_installed()
    dates = pd.date_range("2018-01-31", periods=36, freq="ME").delete(15)
    values = [float(np.sin(2 * np.pi * i / 12) * 10 + i * 0.5 + 50) for i in range(len(dates))]
    frame = pd.DataFrame({"Date": dates, "Value": values})
    result = forecast_time_series(frame, periods=6, model="auto")
    assert result["status"] == "ok", result
    assert result["metrics"]["model"] != "moving_average"
    for row in result["tables"][0]["rows"]:
        assert pd.Timestamp(row[0]).day >= 28
    assert "Dates are irregular, so the forecast assumes equal spacing between observations." in result["flags"]


def test_gapped_daily_is_not_a_flat_average():
    _require_statsmodels_installed()
    dates = pd.date_range("2024-01-01", periods=60, freq="D").delete(25)
    values = [float(10.0 + (i % 7) * 3.0 + i * 0.1) for i in range(len(dates))]
    frame = pd.DataFrame({"Date": dates, "Value": values})
    result = forecast_time_series(frame, periods=7, model="auto")
    assert result["status"] == "ok", result
    assert result["metrics"]["model"] != "moving_average"
    assert "Dates are irregular, so the forecast assumes equal spacing between observations." in result["flags"]


def test_periods_over_50_returns_every_row():
    frame = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=20, freq="D"), "Value": [float(i) for i in range(20)]})
    result = forecast_time_series(frame, periods=100, model="moving_average")
    assert result["status"] == "ok", result
    table = result["tables"][0]
    assert len(table["rows"]) == 100
    assert table["truncated"] is False
    assert table["total_rows"] == 100


def test_explicit_season_too_short_is_flagged():
    frame = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=20, freq="D"), "Value": [float(i) for i in range(20)]})
    result = forecast_time_series(frame, periods=3, model="auto", seasonal_periods=12)
    assert result["status"] == "ok", result
    assert result["metrics"]["model"] != "holt_winters"
    assert any("seasonal_periods=12" in f and "got 20" in f for f in result["flags"])


def test_holt_winters_too_short_is_insufficient_data():
    _require_statsmodels_installed()
    frame = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=20, freq="D"), "Value": [float(i) for i in range(20)]})
    result = forecast_time_series(frame, periods=3, model="holt_winters", seasonal_periods=12)
    assert result["status"] == "error"
    assert result["code"] == "INSUFFICIENT_DATA"
    assert "heuristic method" not in result.get("error", "")


def test_bad_seasonal_period_is_invalid_param():
    frame = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=20, freq="D"), "Value": [float(i) for i in range(20)]})
    res1 = forecast_time_series(frame, seasonal_periods="weekly")
    assert res1["status"] == "error"
    assert res1["code"] == "INVALID_PARAM"
    res2 = run_forecast({"helper": "forecast_time_series", "params": {"seasonal_periods": "weekly"}}, data=frame)
    assert res2["status"] == "error"
    assert res2["code"] == "INVALID_PARAM"
    dec_ok = decompose_time_series(frame, period="7")
    assert dec_ok["status"] == "ok"
    dec_bad = decompose_time_series(frame, period="weekly")
    assert dec_bad["status"] == "error"
    assert dec_bad["code"] == "INVALID_PARAM"


def test_unknown_model_is_invalid_param():
    frame = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=20, freq="D"), "Value": [float(i) for i in range(20)]})
    result = forecast_time_series(frame, model="prophet")
    assert result["status"] == "error"
    assert result["code"] == "INVALID_PARAM"


def test_decompose_nan_cells_are_none():
    _require_statsmodels_installed()
    frame = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=28, freq="D"), "Value": [float(i) for i in range(28)]})
    res = decompose_time_series(frame, period=7)
    assert res["status"] == "ok", res
    table = res["tables"][0]
    cols = table["columns"]
    rows = table["rows"]
    trend_idx = cols.index("trend")
    resid_idx = cols.index("resid")
    assert rows[0][trend_idx] is None
    assert rows[0][resid_idx] is None
    dumped = json.dumps(rows, allow_nan=False, default=str)
    assert dumped is not None


def test_unparseable_dates_are_counted():
    rows = [{"Date": f"2024-01-{day:02d}", "Value": float(day)} for day in range(1, 11)]
    rows.append({"Date": "not-a-date", "Value": 99.0})
    frame = pd.DataFrame(rows)
    res = forecast_time_series(frame, periods=3, model="moving_average")
    assert res["status"] == "ok", res
    assert "Dropped 1 rows with an unparseable date or value" in res["flags"]


def test_moving_average_states_no_interval():
    frame = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=10, freq="D"), "Value": [float(i) for i in range(10)]})
    res = forecast_time_series(frame, periods=3, model="moving_average")
    assert res["status"] == "ok", res
    assert "moving_average has no prediction interval" in res["flags"]


def test_anomaly_tables_use_200_rows():
    _require_statsmodels_installed()
    frame = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=60, freq="D"), "Value": [float(i % 7) for i in range(60)]})
    res = anomaly_detection_time_series(frame, period=7, include_all=True)
    assert res["status"] == "ok", res
    all_scores = next(t for t in res["tables"] if t["name"] == "all_scores")
    assert all_scores["truncated"] is False
    assert len(all_scores["rows"]) == 60


