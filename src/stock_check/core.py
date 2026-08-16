"""Kernfunktionen des Aktien-Analyse-Assistenten.

Jede Funktion ist isoliert testbar (Design-Dokument, Architektur #1).
fetch_price_data(), fetch_fundamentals() und call_llm() sprechen externe
APIs an und sind bewusst nicht unit-getestet (Test-Review #2) -- die
uebrigen Funktionen sind deterministisch und haben pytest-Coverage.
"""

import os
import re
import time

import anthropic
import bcrypt
import openai
import pandas as pd
import yfinance as yf
from google import genai
from supabase import create_client

TICKER_PATTERN = re.compile(r"^[A-Za-z0-9.\-]{1,15}$")

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

# Referenzbereiche wurden im T0-Prototyp-Testlauf noetig: die KI bewertete
# einen niedrigen Verschuldungsgrad (6,56) faelschlich als "hoch", bis diese
# Schwellenwerte explizit mitgegeben wurden.
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

MIN_ANTWORT_LAENGE = 50
MAX_RETRIES = 2
RETRY_PAUSE_SEKUNDEN = 2
# Ohne explizites Timeout kann ein haengender Provider (kein Fehler, keine
# Antwort) den kompletten Multi-KI-Aufruf unbegrenzt blockieren -- die
# Retry-Logik greift erst NACH einer Exception, nie bei einem reinen Haenger.
LLM_TIMEOUT_SEKUNDEN = 30.0


class StockCheckError(Exception):
    """Fehler-Kontrakt (Code-Qualitaet #1): <was> -- <warum>."""

    def __init__(self, was: str, warum: str):
        self.was = was
        self.warum = warum
        super().__init__(f"{was} -- {warum}")


def validate_ticker(ticker: str) -> str:
    """Test-Review #1: schuetzt vor Path-Traversal, bevor der Ticker in
    einen Dateipfad einfliesst (z.B. '../etc', 'AAPL/../x', '..').
    ".." wird trotz erlaubtem Punkt-Zeichen explizit abgelehnt, da echte
    Ticker nie zwei aufeinanderfolgende Punkte enthalten."""
    if not TICKER_PATTERN.match(ticker) or ".." in ticker:
        raise StockCheckError(
            f"Ungueltiger Ticker '{ticker}'",
            "nur Buchstaben, Zahlen, Punkt und Bindestrich erlaubt (max. 15 Zeichen, kein '..')",
        )
    return ticker


def fetch_price_data(ticker: str) -> pd.DataFrame:
    try:
        price_data = yf.Ticker(ticker).history(period="2y")
    except Exception as e:
        # yfinance wirft eigene Exceptions (z.B. YFRateLimitError bei zu
        # vielen Anfragen), die unbehandelt die ganze App abstuerzen liessen
        # statt eine saubere Fehlermeldung zu zeigen -- daher hier gefangen
        # und in den Fehler-Kontrakt uebersetzt.
        raise StockCheckError(
            f"Kursdaten fuer '{ticker}' konnten nicht geladen werden",
            f"yfinance-Fehler: {e}",
        ) from e
    if price_data.empty:
        raise StockCheckError(
            f"Keine Kursdaten fuer '{ticker}'",
            "Ticker existiert nicht oder yfinance liefert keine Daten",
        )
    return price_data


def build_chart_data(price_data: pd.DataFrame) -> pd.DataFrame:
    """Kursverlauf + SMA50/SMA200 als Zeitreihe fuer die Streamlit-Chart-
    Darstellung -- separat von compute_indicators(), da dort nur der jeweils
    letzte Wert gebraucht wird, hier aber der volle Verlauf."""
    close = price_data["Close"]
    chart_data = pd.DataFrame({"Kurs": close})
    if len(close) >= 50:
        chart_data["SMA50"] = close.rolling(50).mean()
    if len(close) >= 200:
        chart_data["SMA200"] = close.rolling(200).mean()
    return chart_data


def compute_indicators(price_data: pd.DataFrame) -> dict:
    """Architektur #2: fehlt fuer einen Indikator ausreichend Historie
    (z.B. SMA200 bei jungen Aktien), wird NUR dieser Indikator uebersprungen,
    nicht der gesamte Durchlauf abgebrochen."""
    close = price_data["Close"]
    indicators = {}

    indicators["SMA50"] = (
        round(close.rolling(50).mean().iloc[-1], 2)
        if len(close) >= 50
        else "nicht verfuegbar, <50 Tage Historie"
    )
    indicators["SMA200"] = (
        round(close.rolling(200).mean().iloc[-1], 2)
        if len(close) >= 200
        else "nicht verfuegbar, <200 Tage Historie"
    )

    if len(close) >= 15:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators["RSI14"] = round(rsi.iloc[-1], 2)
    else:
        indicators["RSI14"] = "nicht verfuegbar, <15 Tage Historie"

    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        indicators["MACD"] = round(macd_line.iloc[-1], 2)
        indicators["MACD_Signal"] = round(signal_line.iloc[-1], 2)
    else:
        indicators["MACD"] = "nicht verfuegbar, <26 Tage Historie"
        indicators["MACD_Signal"] = "nicht verfuegbar, <26 Tage Historie"

    indicators["52W_Hoch"] = round(close.tail(252).max(), 2)
    indicators["52W_Tief"] = round(close.tail(252).min(), 2)
    indicators["20T_Hoch"] = round(close.tail(min(20, len(close))).max(), 2)
    indicators["20T_Tief"] = round(close.tail(min(20, len(close))).min(), 2)
    indicators["Aktueller_Kurs"] = round(close.iloc[-1], 2)

    volume = price_data["Volume"]
    indicators["Volumen_aktuell"] = int(volume.iloc[-1])
    indicators["Volumen_20T_Durchschnitt"] = int(volume.tail(min(20, len(volume))).mean())

    return indicators


def _fetch_yf_info(ticker: str) -> dict:
    """Gemeinsamer .info-Abruf fuer fetch_exchange_info() und
    fetch_fundamentals() -- faengt yfinance-Exceptions (z.B.
    YFRateLimitError), die sonst unbehandelt die App abstuerzen liessen."""
    try:
        return yf.Ticker(ticker).info
    except Exception as e:
        raise StockCheckError(
            f"Unternehmensdaten fuer '{ticker}' konnten nicht geladen werden",
            f"yfinance-Fehler: {e}",
        ) from e


def fetch_exchange_info(ticker: str) -> dict:
    """Dual-gelistete Aktien (z.B. ADS auf Nasdaq vs. Stammaktie auf
    Euronext) haben unterschiedliche Kurse in unterschiedlicher Waehrung
    fuer denselben Ticker-Namensraum -- ohne diese Info wirkt ein korrekter
    Kurs wie ein Datenfehler, wenn man ihn mit einer anderen Boerse vergleicht."""
    info = _fetch_yf_info(ticker)
    return {
        "handelsplatz": info.get("fullExchangeName", "unbekannt"),
        "waehrung": info.get("currency", "unbekannt"),
    }


def fetch_fundamentals(ticker: str) -> dict:
    info = _fetch_yf_info(ticker)
    return {
        "KGV": info.get("trailingPE", "nicht verfuegbar"),
        "KBV": info.get("priceToBook", "nicht verfuegbar"),
        "KUV": info.get("priceToSalesTrailing12Months", "nicht verfuegbar"),
        "ROE": info.get("returnOnEquity", "nicht verfuegbar"),
        "ROA": info.get("returnOnAssets", "nicht verfuegbar"),
        "Verschuldungsgrad_DebtToEquity": info.get("debtToEquity", "nicht verfuegbar"),
        "Analysten_Kursziel_Durchschnitt": info.get("targetMeanPrice", "nicht verfuegbar"),
        "Naechste_Quartalszahlen": fetch_next_earnings_date(ticker),
    }


def fetch_next_earnings_date(ticker: str) -> str:
    """Termin der naechsten Quartalszahlen -- wichtiger Kontext fuer Timing-
    Fragen (z.B. erhoehte Volatilitaet kurz vor der Veroeffentlichung)."""
    try:
        termine = yf.Ticker(ticker).calendar.get("Earnings Date")
    except Exception:
        return "nicht verfuegbar"
    if not termine:
        return "nicht verfuegbar"
    return str(termine[0])


def build_prompt(ticker: str, indicators: dict, fundamentals: dict) -> str:
    kennzahlen = "\n".join(f"- {k}: {v}" for k, v in {**indicators, **fundamentals}.items())
    return f"""Du bist ein Finanzanalyst. Hier sind berechnete Kennzahlen fuer die Aktie {ticker}:

{kennzahlen}

{REFERENZBEREICHE}

Beantworte auf Basis DIESER Zahlen (nicht aus deinem allgemeinen Wissen ueber das Unternehmen) den folgenden Fragenkatalog:

{FRAGENKATALOG}

Antworte strukturiert, eine kurze Antwort pro Frage."""


def _call_llm_anthropic(prompt: str) -> str:
    client = anthropic.Anthropic(timeout=LLM_TIMEOUT_SEKUNDEN)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_llm_gemini(prompt: str) -> str:
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options={"timeout": int(LLM_TIMEOUT_SEKUNDEN * 1000)},
    )
    response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
    return response.text


def _call_llm_openrouter(prompt: str, model: str) -> str:
    client = openai.OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        timeout=LLM_TIMEOUT_SEKUNDEN,
    )
    response = client.chat.completions.create(
        model=model,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# Multi-KI-Vergleich (TODO 1 / Approach B, freie Frage): jeder Eintrag ist
# unabhaengig -- ein fehlender Key oder ein Fehler bei einem Provider stoppt
# nicht die anderen. Reihenfolge hier ist auch die Fallback-Reihenfolge fuer
# call_llm() (Haupt-Analyse, eine KI-Antwort).
PROVIDERS = [
    {
        "id": "anthropic",
        "name": "Claude (Anthropic)",
        "env_key": "ANTHROPIC_API_KEY",
        # Lambda statt direkter Funktionsreferenz: bindet _call_llm_anthropic
        # erst beim Aufruf, nicht beim Modul-Import -- sonst greifen
        # unittest.mock-Patches auf die frueh gebundene Referenz nicht.
        "call": lambda prompt: _call_llm_anthropic(prompt),
    },
    {
        "id": "gemini",
        "name": "Gemini (Google)",
        "env_key": "GEMINI_API_KEY",
        "call": lambda prompt: _call_llm_gemini(prompt),
    },
    {
        "id": "openrouter-nvidia",
        "name": "Nemotron Ultra (NVIDIA, via OpenRouter)",
        "env_key": "OPENROUTER_API_KEY",
        "call": lambda prompt: _call_llm_openrouter(prompt, "nvidia/nemotron-3-ultra-550b-a55b:free"),
    },
    {
        "id": "openrouter-openai",
        "name": "GPT-OSS-20B (OpenAI, via OpenRouter)",
        "env_key": "OPENROUTER_API_KEY",
        "call": lambda prompt: _call_llm_openrouter(prompt, "openai/gpt-oss-20b:free"),
    },
]


def _call_with_retry(call_fn) -> str:
    """Retry bei transienten Fehlern (Outside Voice #5), Plausibilitaets-
    pruefung der Antwort (Outside Voice #4). call_fn: Callable[[], str]."""
    antwort = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            antwort = call_fn()
            break
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_PAUSE_SEKUNDEN)
                continue
            raise StockCheckError(
                "KI-API-Aufruf fehlgeschlagen",
                f"nach {MAX_RETRIES + 1} Versuchen: {e}",
            ) from e

    if not antwort or len(antwort.strip()) < MIN_ANTWORT_LAENGE:
        raise StockCheckError(
            "KI-Antwort unbrauchbar",
            f"leer oder zu kurz ({len(antwort.strip()) if antwort else 0} Zeichen)",
        )
    return antwort


def call_llm(prompt: str) -> str:
    """Haupt-Analyse (Approach A, eine KI-Antwort). LLM_PROVIDER erzwingt
    optional einen einzelnen Provider (z.B. fuer T0-Tests); ist es NICHT
    gesetzt, wird die PROVIDERS-Reihenfolge automatisch durchprobiert --
    schlaegt der erste konfigurierte Provider fehl (z.B. Anthropic ohne
    Guthaben), verhindert das nicht mehr die gesamte Ticker-Analyse."""
    forcierter_provider = os.environ.get("LLM_PROVIDER")
    if forcierter_provider:
        provider = next((p for p in PROVIDERS if p["id"] == forcierter_provider), None)
        if provider is None:
            raise StockCheckError(
                "Unbekannter LLM_PROVIDER",
                f"'{forcierter_provider}' ist keiner der bekannten Provider ({[p['id'] for p in PROVIDERS]})",
            )
        return _call_with_retry(lambda: provider["call"](prompt))

    letzter_fehler: StockCheckError | None = None
    versucht = False
    for provider in PROVIDERS:
        if not os.environ.get(provider["env_key"]):
            continue
        versucht = True
        try:
            return _call_with_retry(lambda p=provider: p["call"](prompt))
        except StockCheckError as e:
            letzter_fehler = e
            continue

    if not versucht:
        raise StockCheckError(
            "Kein KI-Provider konfiguriert",
            "keiner der API-Keys (ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY) ist gesetzt",
        )
    raise letzter_fehler


def ask_all_providers(prompt: str) -> list[dict]:
    """Schickt denselben Prompt an alle konfigurierten Provider (Retry +
    Plausibilitaetspruefung pro Provider ueber _call_with_retry). Gibt fuer
    JEDEN Provider ein Ergebnis zurueck -- nie eine Exception, damit ein
    fehlender Key oder ein Fehler bei einem Provider die anderen nicht
    verhindert."""
    ergebnisse = []
    for provider in PROVIDERS:
        eintrag = {"id": provider["id"], "name": provider["name"]}
        if not os.environ.get(provider["env_key"]):
            eintrag["status"] = "nicht_konfiguriert"
            eintrag["antwort"] = None
            eintrag["fehler"] = f"{provider['env_key']} nicht in .env gesetzt"
        else:
            try:
                eintrag["antwort"] = _call_with_retry(lambda p=provider: p["call"](prompt))
                eintrag["status"] = "ok"
                eintrag["fehler"] = None
            except StockCheckError as e:
                eintrag["status"] = "fehler"
                eintrag["antwort"] = None
                eintrag["fehler"] = f"{e.was} -- {e.warum}"
        ergebnisse.append(eintrag)
    return ergebnisse


def build_freeform_prompt(ticker: str, indicators: dict, fundamentals: dict, frage: str) -> str:
    """Wie build_prompt(), aber mit einer freien Nutzerfrage statt des
    festen Fragenkatalogs -- Approach A (fester Katalog) bleibt unveraendert
    bestehen, das ist eine additive Erweiterung."""
    kennzahlen = "\n".join(f"- {k}: {v}" for k, v in {**indicators, **fundamentals}.items())
    return f"""Du bist ein Finanzanalyst. Hier sind berechnete Kennzahlen fuer die Aktie {ticker}:

{kennzahlen}

{REFERENZBEREICHE}

Beantworte auf Basis DIESER Zahlen (nicht aus deinem allgemeinen Wissen ueber das Unternehmen) folgende Frage:

{frage}"""


def write_output(ticker: str, datum: str, content: str, output_dir: str = "output") -> str:
    """Legt das Ausgabeverzeichnis bei Bedarf an, faengt Schreibfehler ab
    (Failure Mode #1 / Issue 6)."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{ticker}-{datum}.md")
        with open(path, "w") as f:
            f.write(content)
        return path
    except OSError as e:
        raise StockCheckError(
            f"Konnte Ausgabedatei fuer '{ticker}' nicht schreiben",
            str(e),
        ) from e


# --- Accounts + Watchlist (Supabase) ---
#
# Eigene Registrierung/Login statt eines einzelnen geteilten Passworts, damit
# jede Person ihre eigene, dauerhafte Watchlist hat (ueberlebt Streamlit-
# Cloud-Neustarts, im Gegensatz zu einer lokalen Datei oder Browser-
# localStorage). Registrierung braucht zusaetzlich das bestehende
# Einladungswort (REGISTRATION_INVITE_CODE), damit nicht jeder x-beliebige
# Besucher sich selbst einen Account anlegen und die KI-Kontingente
# verbrauchen kann.


def _get_supabase_client():
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def register_user(username: str, password: str, invite_code: str) -> dict:
    username = username.strip()
    erwarteter_code = os.environ.get("REGISTRATION_INVITE_CODE")
    if erwarteter_code and invite_code != erwarteter_code:
        raise StockCheckError("Registrierung fehlgeschlagen", "Einladungscode ist falsch")
    if not username or not password:
        raise StockCheckError(
            "Registrierung fehlgeschlagen", "Benutzername und Passwort duerfen nicht leer sein"
        )

    client = _get_supabase_client()
    vorhanden = client.table("users").select("id").eq("username", username).execute()
    if vorhanden.data:
        raise StockCheckError(
            "Registrierung fehlgeschlagen", f"Benutzername '{username}' ist bereits vergeben"
        )

    ergebnis = client.table("users").insert(
        {"username": username, "password_hash": hash_password(password)}
    ).execute()
    return ergebnis.data[0]


def login_user(username: str, password: str) -> dict:
    client = _get_supabase_client()
    ergebnis = client.table("users").select("*").eq("username", username.strip()).execute()
    # Dieselbe generische Fehlermeldung fuer "Nutzer existiert nicht" und
    # "Passwort falsch" -- verhindert, dass sich existierende Benutzernamen
    # erraten lassen.
    if not ergebnis.data or not verify_password(password, ergebnis.data[0]["password_hash"]):
        raise StockCheckError("Anmeldung fehlgeschlagen", "Benutzername oder Passwort falsch")
    return ergebnis.data[0]


def get_watchlist(user_id: str) -> list[str]:
    client = _get_supabase_client()
    ergebnis = (
        client.table("watchlist")
        .select("ticker")
        .eq("user_id", user_id)
        .order("added_at")
        .execute()
    )
    return [zeile["ticker"] for zeile in ergebnis.data]


def add_to_watchlist(user_id: str, ticker: str) -> None:
    client = _get_supabase_client()
    try:
        client.table("watchlist").insert({"user_id": user_id, "ticker": ticker}).execute()
    except Exception as e:
        # Ticker ist schon in der Watchlist (unique-Constraint) -- kein Fehler,
        # einfach idempotent behandeln.
        if "duplicate" in str(e).lower() or "unique" in str(e).lower():
            return
        raise StockCheckError(f"Konnte '{ticker}' nicht zur Watchlist hinzufuegen", str(e)) from e


def remove_from_watchlist(user_id: str, ticker: str) -> None:
    client = _get_supabase_client()
    client.table("watchlist").delete().eq("user_id", user_id).eq("ticker", ticker).execute()
