"""CLI-Einstiegspunkt. Orchestriert core.py, setzt den Fehler-Kontrakt durch
(Code-Qualitaet #1): Fehler nach stderr im Format 'Fehler: <was> -- <warum>',
Exit-Code 1. Erfolgreicher Lauf: Exit-Code 0."""

import sys
from datetime import date

from dotenv import load_dotenv

from stock_check.core import (
    StockCheckError,
    build_prompt,
    call_llm,
    compute_indicators,
    fetch_exchange_info,
    fetch_fundamentals,
    fetch_price_data,
    validate_ticker,
    write_output,
)

load_dotenv()


def run_for_ticker(ticker: str) -> str:
    ticker = validate_ticker(ticker)
    price_data = fetch_price_data(ticker)
    indicators = compute_indicators(price_data)
    fundamentals = fetch_fundamentals(ticker)
    exchange_info = fetch_exchange_info(ticker)
    # Handelsplatz/Waehrung mitgeben: dual-gelistete Aktien (z.B. ADS auf
    # Nasdaq vs. Stammaktie auf Euronext) haben unterschiedliche Kurse fuer
    # denselben Firmennamen -- ohne diese Info wirkt ein korrekter Kurs wie
    # ein Datenfehler beim Vergleich mit einer anderen Boerse.
    fundamentals_mit_boerse = {
        "Handelsplatz": exchange_info["handelsplatz"],
        "Waehrung": exchange_info["waehrung"],
        **fundamentals,
    }
    prompt = build_prompt(ticker, indicators, fundamentals_mit_boerse)
    antwort = call_llm(prompt)

    content = f"""# {ticker} -- {date.today().isoformat()}

## Kennzahlen

{chr(10).join(f"- {k}: {v}" for k, v in {**indicators, **fundamentals_mit_boerse}.items())}

## KI-Einschaetzung

{antwort}
"""
    return write_output(ticker, date.today().isoformat(), content)


def main() -> None:
    tickers = sys.argv[1:]
    if not tickers:
        print("Nutzung: stock-check TICKER [TICKER ...]", file=sys.stderr)
        sys.exit(1)

    exit_code = 0
    for ticker in tickers:
        try:
            path = run_for_ticker(ticker)
            print(f"{ticker}: geschrieben nach {path}")
        except StockCheckError as e:
            print(f"Fehler: {e.was} -- {e.warum}", file=sys.stderr)
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
