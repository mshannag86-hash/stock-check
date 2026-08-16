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
    add_to_watchlist,
    ask_all_providers,
    build_chart_data,
    build_freeform_prompt,
    build_prompt,
    call_llm,
    compute_indicators,
    fetch_exchange_info,
    fetch_fundamentals,
    fetch_price_data,
    get_watchlist,
    login_user,
    register_user,
    remove_from_watchlist,
    validate_ticker,
    write_output,
)

load_dotenv()

# Streamlit Clouds automatische Secrets-zu-Umgebungsvariablen-Uebertragung
# war unzuverlaessig (Anthropic-SDK meldete "kein Key gefunden" trotz
# gesetztem Secret) -- deshalb hier explizit st.secrets nach os.environ
# spiegeln, damit core.py (das bewusst kein Streamlit kennt und nur
# os.environ liest) ueberall konsistent funktioniert. try/except: lokal
# ohne .streamlit/secrets.toml wirft st.secrets einen Fehler, das ist ok.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass

st.set_page_config(page_title="Aktien-Analyse-Assistent", page_icon="\U0001F4C8", layout="centered")

# Zugangs-Gate: eigener Account pro Person statt einem einzelnen geteilten
# Passwort -- jede Person bekommt dadurch eine eigene, dauerhafte Watchlist
# (ueberlebt Streamlit-Cloud-Neustarts im Gegensatz zu einer lokalen Datei
# oder Browser-localStorage). Registrierung braucht zusaetzlich das
# bestehende Einladungswort (REGISTRATION_INVITE_CODE), damit nicht jeder
# x-beliebige Besucher sich selbst einen Account anlegen und die
# KI-Kontingente des Betreibers verbrauchen kann. Nur aktiv, wenn
# SUPABASE_URL konfiguriert ist -- lokale Entwicklung ohne Supabase bleibt
# reibungslos offen.
if os.environ.get("SUPABASE_URL"):
    if "user" not in st.session_state:
        st.session_state.user = None
    if st.session_state.user is None:
        st.title("\U0001F512 Anmelden")
        login_tab, register_tab = st.tabs(["Anmelden", "Registrieren"])

        with login_tab:
            login_name = st.text_input("Benutzername", key="login_name")
            login_pw = st.text_input("Passwort", type="password", key="login_pw")
            if st.button("Anmelden", type="primary", key="login_button"):
                try:
                    st.session_state.user = login_user(login_name, login_pw)
                    st.rerun()
                except StockCheckError as e:
                    st.error(e.warum)

        with register_tab:
            reg_name = st.text_input("Benutzername", key="reg_name")
            reg_pw = st.text_input("Passwort", type="password", key="reg_pw")
            reg_invite = st.text_input("Einladungscode", type="password", key="reg_invite")
            if st.button("Registrieren", type="primary", key="register_button"):
                try:
                    st.session_state.user = register_user(reg_name, reg_pw, reg_invite)
                    st.rerun()
                except StockCheckError as e:
                    st.error(e.warum)

        st.stop()

# Apple-inspiriertes Erscheinungsbild: viel Weissraum, grosse mutige
# Typografie, kaum Rahmen (stattdessen sanfte Schatten), ein einziger
# Akzentton, pillenfoermige Buttons statt Streamlit-Standardlook.
# -apple-system zuerst in der Font-Stack: rendert auf Mac/iOS als echtes
# SF Pro, Inter ist nur der Fallback fuer andere Plattformen.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
    }

    .block-container {
        max-width: 720px;
        padding-top: 4.5rem;
        padding-bottom: 4rem;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: none;
        border-radius: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04);
        padding: 1.3rem 1.2rem 1.1rem 1.2rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.76rem;
        font-weight: 500;
        color: #86868B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        white-space: normal !important;
        overflow: visible !important;
        line-height: 1.3;
    }
    [data-testid="stMetricLabel"] p {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }
    [data-testid="stMetricValue"] {
        font-weight: 600;
        color: #1D1D1F;
        letter-spacing: -0.01em;
        white-space: normal !important;
        overflow: visible !important;
    }
    [data-testid="stMetricValue"] div, [data-testid="stMetricValue"] p {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }

    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
        border-radius: 14px !important;
        border: 1px solid #D2D2D7 !important;
        padding: 0.7rem 1rem !important;
    }

    /* Pillenfoermige Buttons -- Apples Signature-Element */
    .stButton > button {
        border-radius: 980px;
        font-weight: 500;
        padding: 0.55rem 1.7rem;
        border: none;
    }
    .stButton > button[kind="primary"] {
        background: #0071E3;
    }
    .stButton > button[kind="primary"]:hover {
        background: #0077ED;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important;
        border-radius: 20px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        background: #FFFFFF;
    }

    .hero-eyebrow {
        color: #0071E3;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
        margin-bottom: 0.6rem;
        text-align: center;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 700;
        color: #1D1D1F;
        letter-spacing: -0.03em;
        line-height: 1.08;
        margin-bottom: 0.9rem;
        text-align: center;
    }
    .hero-subtitle {
        color: #86868B;
        font-size: 1.2rem;
        font-weight: 400;
        line-height: 1.4;
        margin-bottom: 2.8rem;
        text-align: center;
    }
    .section-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #86868B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 1.6rem 0 0.7rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-eyebrow">Aktien-Analyse-Assistent</div>
    <div class="hero-title">Ticker rein.<br>Klarheit raus.</div>
    <div class="hero-subtitle">Echte Kennzahlen berechnen. KI-Einschätzung erhalten.<br>Statt Tabs zu jonglieren.</div>
    """,
    unsafe_allow_html=True,
)

if os.environ.get("SUPABASE_URL") and st.session_state.get("user"):
    kopf_links, kopf_rechts = st.columns([4, 1])
    with kopf_links:
        st.caption(f"Angemeldet als **{st.session_state.user['username']}**")
    with kopf_rechts:
        if st.button("Abmelden"):
            st.session_state.user = None
            st.rerun()

watchlist_ticker_zum_analysieren = None
if os.environ.get("SUPABASE_URL") and st.session_state.get("user"):
    watchlist = get_watchlist(st.session_state.user["id"])
    if watchlist:
        st.markdown('<div class="section-label">Watchlist</div>', unsafe_allow_html=True)
        for wl_ticker in watchlist:
            wl_spalte_name, wl_spalte_analysieren, wl_spalte_entfernen = st.columns([3, 2, 1])
            with wl_spalte_name:
                st.markdown(f"**{wl_ticker}**")
            with wl_spalte_analysieren:
                if st.button("Analysieren", key=f"wl_analysieren_{wl_ticker}"):
                    watchlist_ticker_zum_analysieren = wl_ticker
            with wl_spalte_entfernen:
                if st.button("✕", key=f"wl_entfernen_{wl_ticker}"):
                    remove_from_watchlist(st.session_state.user["id"], wl_ticker)
                    st.rerun()

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


def berechne_ticker(roh_ticker: str) -> str | None:
    """Holt Daten + KI-Einschaetzung und legt alles in st.session_state ab.
    Reine Rechenarbeit, KEINE Anzeige -- die uebernimmt render_ticker_ergebnis(),
    das bei JEDEM Rerun aus session_state neu zeichnet. Trennung noetig, weil
    Streamlit-Widgets (z.B. der Watchlist-Button) nur interaktiv bleiben, wenn
    sie bei jedem Durchlauf neu erzeugt werden -- nicht nur direkt nach der
    Berechnung (sonst verschwindet z.B. der Watchlist-Klick wirkungslos,
    sobald irgendein anderer Button denselben Rerun ausloest)."""
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

    try:
        indicators = compute_indicators(price_data)
        fundamentals = fetch_fundamentals(ticker)
        exchange_info = fetch_exchange_info(ticker)
        chart_data = build_chart_data(price_data)
    except StockCheckError as e:
        st.error(f"**{ticker}** -- Fehler: {e.was} -- {e.warum}")
        return None

    alle_kennzahlen = {
        "Handelsplatz": exchange_info["handelsplatz"],
        "Waehrung": exchange_info["waehrung"],
        **indicators,
        **fundamentals,
    }

    with st.spinner(f"{ticker}: KI-Einschaetzung wird eingeholt..."):
        try:
            prompt = build_prompt(ticker, indicators, alle_kennzahlen)
            antwort = call_llm(prompt)
        except StockCheckError as e:
            st.error(f"**{ticker}** -- Fehler: {e.was} -- {e.warum}")
            return None

    content = f"""# {ticker} -- {date.today().isoformat()}

## Kennzahlen

{chr(10).join(f"- {k}: {v}" for k, v in alle_kennzahlen.items())}

## KI-Einschaetzung

{antwort}
"""
    try:
        pfad = write_output(ticker, date.today().isoformat(), content)
    except StockCheckError as e:
        pfad = None
        st.warning(f"Ergebnis wird angezeigt, aber nicht gespeichert: {e.was} -- {e.warum}")

    st.session_state.analysierte_ticker[ticker] = {
        "indicators": indicators,
        "fundamentals_mit_boerse": alle_kennzahlen,
        "exchange_info": exchange_info,
        "chart_data": chart_data,
        "antwort": antwort,
        "pfad": pfad,
    }
    return ticker


def render_ticker_ergebnis(ticker: str, daten: dict) -> None:
    """Zeichnet ein bereits berechnetes Ticker-Ergebnis aus session_state --
    wird bei JEDEM Rerun fuer jeden gespeicherten Ticker aufgerufen, damit
    der Watchlist-Button (und alles andere hier) bei jedem Klick interaktiv
    bleibt statt nach dem naechsten Rerun spurlos zu verschwinden."""
    exchange_info = daten["exchange_info"]

    st.subheader(ticker)
    st.caption(
        f"Handelsplatz: {exchange_info['handelsplatz']} | Waehrung: {exchange_info['waehrung']} "
        "-- Achtung bei dual-gelisteten Aktien (z.B. ADS vs. Stammaktie): "
        "andere Boerse kann einen deutlich anderen Kurs zeigen."
    )

    if os.environ.get("SUPABASE_URL") and st.session_state.get("user"):
        user_id = st.session_state.user["id"]
        auf_watchlist = ticker in get_watchlist(user_id)
        if auf_watchlist:
            if st.button("✓ Auf der Watchlist -- entfernen", key=f"wl_toggle_{ticker}"):
                remove_from_watchlist(user_id, ticker)
                st.rerun()
        else:
            if st.button("☆ Zur Watchlist hinzufügen", key=f"wl_toggle_{ticker}"):
                add_to_watchlist(user_id, ticker)
                st.rerun()

    with st.container(border=True):
        st.plotly_chart(
            build_chart_figure(ticker, daten["chart_data"], daten["indicators"]),
            use_container_width=True,
        )
        kennzahlen_spalten = st.columns(3)
        for i, (name, wert) in enumerate(daten["fundamentals_mit_boerse"].items()):
            with kennzahlen_spalten[i % 3]:
                st.metric(label=format_label(name), value=format_kennzahl(name, wert))

    st.markdown('<div class="section-label">KI-Einschätzung</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(daten["antwort"])

    if daten["pfad"]:
        st.caption(f"Gespeichert unter: {daten['pfad']}")


if "analysierte_ticker" not in st.session_state:
    st.session_state.analysierte_ticker = {}

if watchlist_ticker_zum_analysieren:
    berechne_ticker(watchlist_ticker_zum_analysieren)
elif analysieren and ticker_input:
    for roh_ticker in ticker_input.split(","):
        berechne_ticker(roh_ticker)
elif analysieren:
    st.warning("Bitte mindestens einen Ticker eingeben.")

# Rendert ALLE bisher berechneten Ticker bei JEDEM Rerun neu -- nicht nur
# direkt nach der Berechnung -- damit Folge-Interaktionen wie der
# Watchlist-Button zuverlaessig funktionieren (siehe berechne_ticker()).
for ticker, daten in st.session_state.analysierte_ticker.items():
    render_ticker_ergebnis(ticker, daten)
    st.divider()


# --- Eigene Frage an alle konfigurierten KIs (TODO 1 / Approach B) ---

if st.session_state.analysierte_ticker:
    st.markdown('<div class="section-label">Eigene Frage stellen</div>', unsafe_allow_html=True)
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
        daten = st.session_state.analysierte_ticker[gewaehlter_ticker]
        prompt = build_freeform_prompt(
            gewaehlter_ticker, daten["indicators"], daten["fundamentals_mit_boerse"], eigene_frage
        )

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
