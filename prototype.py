"""T0 Wegwerf-Prototyp — testet, ob der Fragenkatalog + berechnete Kennzahlen
brauchbare Claude-Einschaetzungen liefern. Nicht Teil der finalen Struktur
(siehe Design-Dokument, Implementation Task T0). Keine Tests, kein
Fehler-Kontrakt, keine Ticker-Validierung -- das kommt erst nach T0.

Nutzung:
    ANTHROPIC_API_KEY=... uv run prototype.py AAPL MSFT NVDA
"""

import os
import sys

import anthropic
import openai
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from google import genai

load_dotenv()

# T0-Hinweis: Anthropic-Billing des Nutzers war beim Prototyp-Test blockiert
# (Kreditkarte abgelehnt), OpenAI-Billing ebenfalls nicht vorhanden -- Gemini
# (Google AI Studio Free-Tier, kein Kreditkarten-Zwang) ist hier nur ein
# temporaerer Ersatz, um den Fragenkatalog/Prompt-Aufbau zu validieren.
# Finale Wahl laut Design-Dokument bleibt die Claude API, sobald das
# Billing-Problem geloest ist.
PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")

FRAGENKATALOG = """\
1. Wie hat sich der Kurs zuletzt entwickelt (Trend anhand SMA 50 vs. SMA 200 -- Golden Cross / Death Cross)?
2. Ist die Aktie laut RSI(14) aktuell ueberkauft oder ueberverkauft?
3. Was zeigt der MACD ueber Trendstaerke bzw. einen moeglichen Trendwechsel?
4. Wie nah steht der aktuelle Kurs an Unterstuetzung/Widerstand (52-Wochen-Hoch/Tief, 20-Tage-Hoch/Tief)?
5. Bestaetigt das aktuelle Handelsvolumen den Kurstrend, oder laeuft die Bewegung auf schwachem Volumen?
6. Ist die Aktie anhand des KGV im Vergleich zur Branche eher guenstig oder teuer bewertet?
7. Wie steht das KBV -- notiert die Aktie deutlich ueber oder unter ihrem Substanzwert?
8. Falls kein Gewinn ausgewiesen wird: Was sagt das KUV ueber die Bewertung?
9. Wie effizient setzt das Unternehmen Kapital ein (ROE/ROA)?
10. Wie krisenfest ist die Bilanz (Eigenkapitalquote/Verschuldungsgrad)?
11. Basierend auf allen obigen Zahlen: Gesamteinschaetzung -- eher attraktiv, neutral oder eher unattraktiv, und die Kernbegruendung in 2-3 Saetzen?
"""


def compute_indicators(price_data: pd.DataFrame) -> dict:
    close = price_data["Close"]
    indicators = {}

    if len(close) >= 200:
        indicators["SMA50"] = round(close.rolling(50).mean().iloc[-1], 2)
        indicators["SMA200"] = round(close.rolling(200).mean().iloc[-1], 2)
    else:
        indicators["SMA50"] = round(close.rolling(50).mean().iloc[-1], 2) if len(close) >= 50 else "nicht verfuegbar, <50 Tage Historie"
        indicators["SMA200"] = "nicht verfuegbar, <200 Tage Historie"

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    indicators["RSI14"] = round(rsi.iloc[-1], 2) if len(close) >= 15 else "nicht verfuegbar, <15 Tage Historie"

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    indicators["MACD"] = round(macd_line.iloc[-1], 2)
    indicators["MACD_Signal"] = round(signal_line.iloc[-1], 2)

    indicators["52W_Hoch"] = round(close.tail(252).max(), 2)
    indicators["52W_Tief"] = round(close.tail(252).min(), 2)
    indicators["20T_Hoch"] = round(close.tail(20).max(), 2)
    indicators["20T_Tief"] = round(close.tail(20).min(), 2)
    indicators["Aktueller_Kurs"] = round(close.iloc[-1], 2)

    volume = price_data["Volume"]
    indicators["Volumen_aktuell"] = int(volume.iloc[-1])
    indicators["Volumen_20T_Durchschnitt"] = int(volume.tail(20).mean())

    return indicators


def fetch_fundamentals(ticker: str) -> dict:
    info = yf.Ticker(ticker).info
    return {
        "KGV": info.get("trailingPE", "nicht verfuegbar"),
        "KBV": info.get("priceToBook", "nicht verfuegbar"),
        "KUV": info.get("priceToSalesTrailing12Months", "nicht verfuegbar"),
        "ROE": info.get("returnOnEquity", "nicht verfuegbar"),
        "ROA": info.get("returnOnAssets", "nicht verfuegbar"),
        "Verschuldungsgrad_DebtToEquity": info.get("debtToEquity", "nicht verfuegbar"),
    }


REFERENZBEREICHE = """\
Nutze diese groben Referenzbereiche zur Einordnung -- rechne NICHT gegen dein
eigenes, moeglicherweise abweichendes Gefuehl fuer "hoch"/"niedrig":
- RSI(14): <30 ueberverkauft, 30-70 neutral, >70 ueberkauft
- KGV: <15 guenstig, 15-25 moderat, >25 teuer
- KBV: <1 unter Substanzwert, 1-3 moderat, >3 deutlich ueber Substanzwert
- KUV: <1 guenstig, 1-4 moderat, >4 teuer
- ROE: <10% schwach, 10-20% solide, >20% sehr gut
- ROA: <5% schwach, 5-10% solide, >10% sehr gut
- Verschuldungsgrad (Debt-to-Equity): <50 niedrig/krisenfest, 50-100 moderat, >100 hoch/riskant
"""


def build_prompt(ticker: str, indicators: dict, fundamentals: dict) -> str:
    kennzahlen = "\n".join(f"- {k}: {v}" for k, v in {**indicators, **fundamentals}.items())
    return f"""Du bist ein Finanzanalyst. Hier sind berechnete Kennzahlen fuer die Aktie {ticker}:

{kennzahlen}

{REFERENZBEREICHE}

Beantworte auf Basis DIESER Zahlen (nicht aus deinem allgemeinen Wissen ueber das Unternehmen) den folgenden Fragenkatalog:

{FRAGENKATALOG}

Antworte strukturiert, eine kurze Antwort pro Frage."""


def call_llm_anthropic(prompt: str) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_llm_openai(prompt: str) -> str:
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def call_llm_gemini(prompt: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
    )
    return response.text


def call_llm(prompt: str) -> str:
    if PROVIDER == "gemini":
        return call_llm_gemini(prompt)
    if PROVIDER == "openai":
        return call_llm_openai(prompt)
    return call_llm_anthropic(prompt)


def run_for_ticker(ticker: str) -> None:
    print(f"\n{'=' * 60}\n{ticker}\n{'=' * 60}")

    price_data = yf.Ticker(ticker).history(period="2y")
    if price_data.empty:
        print(f"Fehler: keine Kursdaten fuer {ticker} gefunden")
        return

    indicators = compute_indicators(price_data)
    fundamentals = fetch_fundamentals(ticker)
    prompt = build_prompt(ticker, indicators, fundamentals)
    antwort = call_llm(prompt)

    print("\n--- Berechnete Kennzahlen ---")
    for k, v in {**indicators, **fundamentals}.items():
        print(f"{k}: {v}")

    print("\n--- Claude-Einschaetzung ---")
    print(antwort)


def main() -> None:
    tickers = sys.argv[1:]
    if not tickers:
        print("Nutzung: uv run prototype.py TICKER [TICKER ...]")
        sys.exit(1)

    for ticker in tickers:
        run_for_ticker(ticker)


if __name__ == "__main__":
    main()
