"""Streamlit-Oberflaeche fuer den Aktien-Analyse-Assistenten.

Wiederverwendet core.py 1:1 (Fehler-Kontrakt, Ticker-Validierung, Retry,
Plausibilitaetspruefung bleiben unveraendert) -- nur die Ein-/Ausgabe laeuft
jetzt ueber den Browser statt die Kommandozeile.

Start: uv run streamlit run src/stock_check/app.py
"""

import os
from datetime import date

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from stock_check.core import (
    StockCheckError,
    ask_all_providers,
    build_chart_data,
    build_freeform_prompt,
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

st.set_page_config(page_title="Aktien-Analyse-Assistent", page_icon="\U0001F4C8", layout="centered")

# Zugangs-Gate: aktiv NUR wenn APP_PASSWORD gesetzt ist (z.B. beim Deployment
# auf Streamlit Community Cloud) -- lokale Entwicklung ohne gesetztes
# Passwort bleibt reibungslos offen. Verhindert, dass irgendwer mit dem
# oeffentlichen Link die (kostenpflichtigen/limitierten) KI-Kontingente
# des Betreibers verbraucht.
_APP_PASSWORD = os.environ.get("APP_PASSWORD")
if _APP_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.title("\U0001F512 Zugang")
        eingabe = st.text_input("Passwort", type="password")
        if st.button("Anmelden", type="primary"):
            if eingabe == _APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Falsches Passwort.")
        st.stop()

# Modernes Erscheinungsbild: Inter-Schrift, geglaettete Karten fuer
# Kennzahlen/Chart, dezente Rundungen statt Streamlit-Standardlook.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        max-width: 880px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stMetric"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.9rem 1rem 0.7rem 1rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        white-space: normal !important;
        overflow: visible !important;
        line-height: 1.25;
    }
    [data-testid="stMetricLabel"] p {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }
    [data-testid="stMetricValue"] {
        font-weight: 700;
        color: #0F172A;
    }

    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
        border-radius: 10px !important;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.4rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
    }

    .hero-eyebrow {
        color: #2563EB;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #64748B;
        font-size: 1rem;
        margin-bottom: 1.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-eyebrow">📈 Aktien-Analyse-Assistent</div>
    <div class="hero-title">Ticker rein, Klarheit raus.</div>
    <div class="hero-subtitle">Echte Kennzahlen berechnen, KI-Einschätzung auf dieser Basis erhalten — statt Tabs zu jonglieren.</div>
    """,
    unsafe_allow_html=True,
)

ticker_input = st.text_input(
    "Ticker (mehrere durch Komma trennen, z.B. AAPL, MSFT, NVDA)",
    placeholder="AAPL",
)
analysieren = st.button("Analysieren", type="primary")


LABEL_ANZEIGE = {
    "SMA50": "SMA 50",
    "SMA200": "SMA 200",
    "RSI14": "RSI (14)",
    "MACD_Signal": "MACD-Signal",
    "52W_Hoch": "52W-Hoch",
    "52W_Tief": "52W-Tief",
    "20T_Hoch": "20T-Hoch",
    "20T_Tief": "20T-Tief",
    "Aktueller_Kurs": "Kurs",
    "Volumen_aktuell": "Volumen",
    "Volumen_20T_Durchschnitt": "Ø Volumen (20T)",
    "Verschuldungsgrad_DebtToEquity": "Verschuldungsgrad",
    "Analysten_Kursziel_Durchschnitt": "Ø Kursziel (Analysten)",
    "Naechste_Quartalszahlen": "Nächste Quartalszahlen",
    "Waehrung": "Währung",
}


def format_label(name: str) -> str:
    """Technische Feldnamen (z.B. 'Verschuldungsgrad_DebtToEquity') werden
    sonst als abgeschnittene Grossbuchstaben-Wand angezeigt -- lesbarere
    Labels nur fuer die Anzeige, die Datenschluessel selbst bleiben unveraendert."""
    return LABEL_ANZEIGE.get(name, name)


def format_kennzahl(name: str, wert) -> str:
    """Grosse Volumen-Zahlen werden sonst im schmalen Kachel-Layout
    abgeschnitten (z.B. '18607230.00') -- als Millionen-Kurzform lesbarer."""
    if not isinstance(wert, (int, float)):
        return str(wert)
    if "Volumen" in name:
        return f"{wert / 1_000_000:.2f} Mio."
    if isinstance(wert, int):
        return f"{wert:,}".replace(",", ".")
    return f"{wert:.2f}"


def build_chart_figure(ticker: str, chart_data, indicators: dict) -> go.Figure:
    """Interaktiver Kurschart: Kurs + SMA50/SMA200 als Linien, 52-Wochen-
    Hoch/Tief als Referenzlinien -- visualisiert direkt Frage 4 des
    Fragenkatalogs (Naehe zu Unterstuetzung/Widerstand)."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["Kurs"],
            name="Kurs",
            line=dict(color="#2563eb", width=2),
        )
    )
    if "SMA50" in chart_data.columns:
        fig.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["SMA50"],
                name="SMA 50",
                line=dict(color="#f59e0b", width=1.5, dash="dot"),
            )
        )
    if "SMA200" in chart_data.columns:
        fig.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["SMA200"],
                name="SMA 200",
                line=dict(color="#8b5cf6", width=1.5, dash="dash"),
            )
        )

    hoch = indicators.get("52W_Hoch")
    tief = indicators.get("52W_Tief")
    if isinstance(hoch, (int, float)):
        fig.add_hline(
            y=hoch,
            line=dict(color="#ef4444", width=1, dash="dot"),
            annotation_text=f"52W-Hoch {hoch}",
            annotation_position="top right",
        )
    if isinstance(tief, (int, float)):
        fig.add_hline(
            y=tief,
            line=dict(color="#22c55e", width=1, dash="dot"),
            annotation_text=f"52W-Tief {tief}",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=f"{ticker} -- Kursverlauf (2 Jahre)",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=20, l=20, r=20),
        yaxis_title="Kurs",
        hovermode="x unified",
    )
    return fig


def analysiere_ticker(roh_ticker: str) -> tuple[str, dict, dict] | None:
    ticker = roh_ticker.strip()
    if not ticker:
        return None

    with st.spinner(f"{ticker}: Kursdaten werden geladen..."):
        try:
            ticker = validate_ticker(ticker)
            price_data = fetch_price_data(ticker)
        except StockCheckError as e:
            st.error(f"**{ticker}** -- Fehler: {e.was} -- {e.warum}")
            return None

    indicators = compute_indicators(price_data)
    fundamentals = fetch_fundamentals(ticker)
    exchange_info = fetch_exchange_info(ticker)

    st.subheader(ticker)
    st.caption(
        f"Handelsplatz: {exchange_info['handelsplatz']} | Waehrung: {exchange_info['waehrung']} "
        "-- Achtung bei dual-gelisteten Aktien (z.B. ADS vs. Stammaktie): "
        "andere Boerse kann einen deutlich anderen Kurs zeigen."
    )

    with st.container(border=True):
        chart_data = build_chart_data(price_data)
        st.plotly_chart(build_chart_figure(ticker, chart_data, indicators), use_container_width=True)

        alle_kennzahlen = {
            "Handelsplatz": exchange_info["handelsplatz"],
            "Waehrung": exchange_info["waehrung"],
            **indicators,
            **fundamentals,
        }
        kennzahlen_spalten = st.columns(3)
        for i, (name, wert) in enumerate(alle_kennzahlen.items()):
            with kennzahlen_spalten[i % 3]:
                st.metric(label=format_label(name), value=format_kennzahl(name, wert))

    with st.spinner(f"{ticker}: KI-Einschaetzung wird eingeholt..."):
        try:
            fundamentals_mit_boerse = {
                "Handelsplatz": exchange_info["handelsplatz"],
                "Waehrung": exchange_info["waehrung"],
                **fundamentals,
            }
            prompt = build_prompt(ticker, indicators, fundamentals_mit_boerse)
            antwort = call_llm(prompt)
        except StockCheckError as e:
            st.error(f"**{ticker}** -- Fehler: {e.was} -- {e.warum}")
            return None

    st.markdown("#### 🤖 KI-Einschätzung")
    with st.container(border=True):
        st.markdown(antwort)

    content = f"""# {ticker} -- {date.today().isoformat()}

## Kennzahlen

{chr(10).join(f"- {k}: {v}" for k, v in alle_kennzahlen.items())}

## KI-Einschaetzung

{antwort}
"""
    try:
        pfad = write_output(ticker, date.today().isoformat(), content)
        st.caption(f"Gespeichert unter: {pfad}")
    except StockCheckError as e:
        st.warning(f"Ergebnis wurde angezeigt, aber nicht gespeichert: {e.was} -- {e.warum}")

    return ticker, indicators, fundamentals_mit_boerse


if "analysierte_ticker" not in st.session_state:
    st.session_state.analysierte_ticker = {}

if analysieren and ticker_input:
    for roh_ticker in ticker_input.split(","):
        ergebnis = analysiere_ticker(roh_ticker)
        if ergebnis:
            ticker, indicators, fundamentals_mit_boerse = ergebnis
            st.session_state.analysierte_ticker[ticker] = (indicators, fundamentals_mit_boerse)
        st.divider()
elif analysieren:
    st.warning("Bitte mindestens einen Ticker eingeben.")


# --- Eigene Frage an alle konfigurierten KIs (TODO 1 / Approach B) ---

if st.session_state.analysierte_ticker:
    st.divider()
    st.markdown("### 💬 Eigene Frage stellen")
    st.caption(
        "Wird zusammen mit denselben berechneten Kennzahlen an alle konfigurierten KIs geschickt "
        "(Claude, Gemini, sowie zwei Modelle ueber OpenRouter). Fehlt ein API-Key oder schlaegt "
        "ein Provider fehl, wird das klar angezeigt -- die anderen laufen trotzdem weiter."
    )

    gewaehlter_ticker = st.selectbox(
        "Fuer welchen Ticker?", options=list(st.session_state.analysierte_ticker.keys())
    )
    eigene_frage = st.text_area(
        "Deine Frage",
        placeholder="z.B. Wie schaetzt du das Chance-Risiko-Verhaeltnis fuer einen Einstieg diese Woche ein?",
    )
    frage_stellen = st.button("An alle KIs schicken", type="primary")

    if frage_stellen and eigene_frage.strip():
        indicators, fundamentals_mit_boerse = st.session_state.analysierte_ticker[gewaehlter_ticker]
        prompt = build_freeform_prompt(gewaehlter_ticker, indicators, fundamentals_mit_boerse, eigene_frage)

        with st.spinner("Frage wird an alle KIs geschickt..."):
            ergebnisse = ask_all_providers(prompt)

        tabs = st.tabs([e["name"] for e in ergebnisse])
        for tab, eintrag in zip(tabs, ergebnisse):
            with tab:
                if eintrag["status"] == "ok":
                    st.markdown(eintrag["antwort"])
                elif eintrag["status"] == "nicht_konfiguriert":
                    st.info(f"Nicht konfiguriert: {eintrag['fehler']}")
                else:
                    st.error(f"Fehler: {eintrag['fehler']}")
    elif frage_stellen:
        st.warning("Bitte eine Frage eingeben.")
