from unittest.mock import patch

import pandas as pd
import pytest

from stock_check.core import (
    PROVIDERS,
    StockCheckError,
    ask_all_providers,
    build_chart_data,
    build_prompt,
    call_llm,
    compute_indicators,
    validate_ticker,
    write_output,
)


# --- validate_ticker() -- Test-Review #1 (Path-Traversal-Schutz) ---


def test_validate_ticker_accepts_valid_format():
    assert validate_ticker("AAPL") == "AAPL"
    assert validate_ticker("BRK.B") == "BRK.B"
    assert validate_ticker("7203.T") == "7203.T"


@pytest.mark.parametrize("bad_ticker", ["../etc", "AAPL/../x", "AAPL/passwd", "..", "a/b", ""])
def test_validate_ticker_rejects_path_traversal(bad_ticker):
    with pytest.raises(StockCheckError):
        validate_ticker(bad_ticker)


# --- compute_indicators() -- Architektur #2 (fehlende Historie) ---


def _make_price_data(n_days: int) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "Close": [100.0 + i * 0.1 for i in range(n_days)],
            "Volume": [1_000_000 + i * 1000 for i in range(n_days)],
        },
        index=dates,
    )


def test_compute_indicators_with_sufficient_history():
    price_data = _make_price_data(300)
    indicators = compute_indicators(price_data)

    assert isinstance(indicators["SMA50"], float)
    assert isinstance(indicators["SMA200"], float)
    assert isinstance(indicators["RSI14"], float)
    assert isinstance(indicators["MACD"], float)


def test_compute_indicators_skips_sma200_when_insufficient_history():
    price_data = _make_price_data(100)
    indicators = compute_indicators(price_data)

    assert indicators["SMA200"] == "nicht verfuegbar, <200 Tage Historie"
    assert isinstance(indicators["SMA50"], float)


def test_compute_indicators_skips_all_when_very_short_history():
    price_data = _make_price_data(10)
    indicators = compute_indicators(price_data)

    assert indicators["SMA50"] == "nicht verfuegbar, <50 Tage Historie"
    assert indicators["RSI14"] == "nicht verfuegbar, <15 Tage Historie"
    assert indicators["MACD"] == "nicht verfuegbar, <26 Tage Historie"
    # aktueller Kurs bleibt trotzdem berechenbar
    assert isinstance(indicators["Aktueller_Kurs"], float)


# --- build_chart_data() ---


def test_build_chart_data_includes_close_price():
    price_data = _make_price_data(30)
    chart_data = build_chart_data(price_data)
    assert "Kurs" in chart_data.columns
    assert len(chart_data) == 30


def test_build_chart_data_omits_sma_when_insufficient_history():
    price_data = _make_price_data(30)
    chart_data = build_chart_data(price_data)
    assert "SMA50" not in chart_data.columns
    assert "SMA200" not in chart_data.columns


def test_build_chart_data_includes_sma50_with_enough_history():
    price_data = _make_price_data(100)
    chart_data = build_chart_data(price_data)
    assert "SMA50" in chart_data.columns
    assert "SMA200" not in chart_data.columns


# --- build_prompt() ---


def test_build_prompt_includes_all_fields():
    prompt = build_prompt("AAPL", {"SMA50": 100.0}, {"KGV": 20.0})
    assert "AAPL" in prompt
    assert "SMA50: 100.0" in prompt
    assert "KGV: 20.0" in prompt


def test_build_prompt_includes_reference_ranges():
    prompt = build_prompt("AAPL", {}, {})
    assert "Referenzbereiche" in prompt or "Verschuldungsgrad" in prompt


def test_build_prompt_marks_missing_indicator():
    prompt = build_prompt("XYZ", {"SMA200": "nicht verfuegbar, <200 Tage Historie"}, {})
    assert "nicht verfuegbar" in prompt


# --- call_llm() -- Outside Voice #4 (Plausibilitaet) + #5 (Retry) ---


def test_call_llm_raises_on_empty_response():
    with patch("stock_check.core._call_llm_once", return_value=""):
        with pytest.raises(StockCheckError, match="unbrauchbar"):
            call_llm("irgendein prompt")


def test_call_llm_raises_on_too_short_response():
    with patch("stock_check.core._call_llm_once", return_value="ok"):
        with pytest.raises(StockCheckError, match="unbrauchbar"):
            call_llm("irgendein prompt")


def test_call_llm_returns_valid_response():
    lange_antwort = "x" * 100
    with patch("stock_check.core._call_llm_once", return_value=lange_antwort):
        assert call_llm("irgendein prompt") == lange_antwort


def test_call_llm_retries_on_transient_error_then_succeeds():
    lange_antwort = "x" * 100
    with patch(
        "stock_check.core._call_llm_once",
        side_effect=[Exception("503 UNAVAILABLE"), lange_antwort],
    ):
        with patch("stock_check.core.time.sleep"):
            assert call_llm("irgendein prompt") == lange_antwort


def test_call_llm_gives_up_after_max_retries():
    with patch("stock_check.core._call_llm_once", side_effect=Exception("503 UNAVAILABLE")):
        with patch("stock_check.core.time.sleep"):
            with pytest.raises(StockCheckError, match="fehlgeschlagen"):
                call_llm("irgendein prompt")


# --- ask_all_providers() -- Multi-KI (TODO 1 / Approach B, freie Frage) ---


def test_ask_all_providers_marks_missing_env_key_without_calling():
    with patch("stock_check.core.os.environ", {}):
        ergebnisse = ask_all_providers("irgendein prompt")

    assert len(ergebnisse) == len(PROVIDERS)
    for eintrag in ergebnisse:
        assert eintrag["status"] == "nicht_konfiguriert"
        assert eintrag["antwort"] is None


def test_ask_all_providers_isolates_failing_provider():
    lange_antwort = "x" * 100
    alle_env_keys = {p["env_key"]: "dummy" for p in PROVIDERS}

    def fake_call_with_retry(call_fn):
        # simuliert: erster Provider schlaegt fehl, alle anderen liefern eine Antwort
        if fake_call_with_retry.aufrufe == 0:
            fake_call_with_retry.aufrufe += 1
            raise StockCheckError("KI-API-Aufruf fehlgeschlagen", "simulierter Fehler")
        return lange_antwort

    fake_call_with_retry.aufrufe = 0

    with patch("stock_check.core.os.environ", alle_env_keys):
        with patch("stock_check.core._call_with_retry", side_effect=fake_call_with_retry):
            ergebnisse = ask_all_providers("irgendein prompt")

    assert ergebnisse[0]["status"] == "fehler"
    assert ergebnisse[0]["antwort"] is None
    for eintrag in ergebnisse[1:]:
        assert eintrag["status"] == "ok"
        assert eintrag["antwort"] == lange_antwort


# --- write_output() -- Failure Mode #1 / Issue 6 ---


def test_write_output_creates_directory_and_file(tmp_path):
    output_dir = str(tmp_path / "output")
    path = write_output("AAPL", "2026-08-14", "Testinhalt", output_dir=output_dir)

    assert path == f"{output_dir}/AAPL-2026-08-14.md"
    with open(path) as f:
        assert f.read() == "Testinhalt"


def test_write_output_overwrites_existing_file(tmp_path):
    output_dir = str(tmp_path / "output")
    write_output("AAPL", "2026-08-14", "Erster Inhalt", output_dir=output_dir)
    path = write_output("AAPL", "2026-08-14", "Zweiter Inhalt", output_dir=output_dir)

    with open(path) as f:
        assert f.read() == "Zweiter Inhalt"


def test_write_output_raises_stockcheckerror_on_permission_denied(tmp_path):
    output_dir = str(tmp_path / "readonly")
    import os

    os.makedirs(output_dir)
    os.chmod(output_dir, 0o444)
    try:
        with pytest.raises(StockCheckError):
            write_output("AAPL", "2026-08-14", "Inhalt", output_dir=output_dir)
    finally:
        os.chmod(output_dir, 0o755)
