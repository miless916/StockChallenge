"""
AI Stock Picking Challenge - Full Automation (v8)
------------------------------------------------------------------------
Each time you run this script, it:
  1. Asks Claude, ChatGPT, and Gemini for 5 stock picks each, with a
     1-10 confidence rating for each pick.
  2. Parses each response into (ticker, confidence) pairs.
  3. Fills in "Day 1" (next trading day close) prices for any picks
     from a previous run that are still pending.
  4. Logs today's picks with today's closing price as "Day 0" into
     stock_challenge_log.csv (detailed, one row per stock per AI).
  5. Whenever a prior day's picks for an AI just got their Day 1 prices
     filled in, computes that day's average return across its picks
     and updates a running theoretical portfolio value for that AI
     (starting at $1,000), saved to portfolio_value_log.csv.

Run this once per day, ideally right around market close.

SETUP (one-time):
    pip install anthropic openai google-genai yfinance pandas openpyxl python-dotenv

    .env file in this folder with:
        ANTHROPIC_API_KEY=sk-ant-...
        OPENAI_API_KEY=sk-...
        GEMINI_API_KEY=AIza...

NOTE ON v8 CHANGES:
  - New prompt: no Magnificent 7 stocks, and the models are encouraged
    (not required) to diversify across sectors. If a model picks a Mag 7 stock anyway,
    it is still logged (its call, its choice) — the run output just
    flags it so you can see how often the rule gets ignored.
  - New "Prompt_Version" column in BOTH CSVs. Rows logged before v8
    are back-filled as "v1_open" (the original unconstrained prompt);
    new rows are "v2_no_mag7_diversify". Filter or pivot on this column
    to compare the two eras. Portfolio values keep compounding from
    where they left off.
  - Parser now handles class-share tickers like BRK.B (converted to
    BRK-B for yfinance), which are more likely now that mega-caps are
    excluded.
  - Fixed same-day duplicate check: it was keyed on ticker only, so on
    a re-run the same day, one AI's pick could block another AI's
    identical pick. It's now keyed on (Source, Ticker).
"""

import os
import re
import time
from datetime import datetime

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ---- Model choices ----
CLAUDE_MODEL = "claude-sonnet-5"
OPENAI_MODEL = "gpt-5.4"
GEMINI_MODEL_PRIMARY = "gemini-flash-latest"
GEMINI_MODEL_FALLBACK = "gemini-3.6-flash"

STARTING_PORTFOLIO_VALUE = 1000.0

# ---- Prompt versioning ----
PROMPT_VERSION = "v2_no_mag7_diversify"
LEGACY_PROMPT_VERSION = "v1_open"   # label applied to rows logged before v8

EXCLUDED_TICKERS = {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA"}

PROMPT = (
    "You are participating in a stock-picking challenge. Based on current "
    "market conditions, pick 5 publicly traded US stocks you believe are "
    "most likely to see their price increase over the next trading day. "
    "RULES: (1) Do NOT pick any of the Magnificent 7 stocks "
    "(AAPL, MSFT, GOOGL/GOOG, AMZN, NVDA, META, TSLA). "
    "(2) You are encouraged to diversify across sectors, but if you "
    "believe a particular sector is likely to rise, you may pick more "
    "than one stock from it. "
    "For each pick, also give a confidence rating from 1 (low) to 10 (high) "
    "reflecting how confident you are in that specific pick. "
    "Respond with ONLY a comma-separated list in this exact format, "
    "nothing else — no explanation, no extra text: "
    "TICKER:CONFIDENCE, TICKER:CONFIDENCE, TICKER:CONFIDENCE, TICKER:CONFIDENCE, TICKER:CONFIDENCE "
    "Example: JPM:7, LLY:6, XOM:5, CAT:8, AVGO:6"
)

DETAIL_FILE = "stock_challenge_log.csv"
DETAIL_COLUMNS = ["Date_Picked", "Source", "Ticker", "Confidence",
                  "Price_Day0", "Price_Day1", "Prompt_Version"]

PORTFOLIO_FILE = "portfolio_value_log.csv"
PORTFOLIO_COLUMNS = ["Date", "Source", "Daily_Return_Pct", "Portfolio_Value", "Prompt_Version"]

TRANSIENT_ERROR_KEYWORDS = ["connection", "timeout", "timed out", "503", "unavailable", "overloaded", "rate limit"]


def _is_transient(error):
    text = str(error).lower()
    return any(keyword in text for keyword in TRANSIENT_ERROR_KEYWORDS)


def _retry_call(fn, label, max_retries=4):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if _is_transient(e):
                wait_seconds = 5 * attempt
                print(f"  {label} transient error (attempt {attempt}/{max_retries}): {e}")
                print(f"  Retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)
            else:
                raise
    raise last_error


# ---------------- AI query functions ----------------

def _describe_exception(e):
    """Build a detailed diagnostic string from an exception, including
    the underlying cause chain, which the SDKs often hide behind a
    generic message like 'Connection error'."""
    parts = [f"{type(e).__name__}: {e}"]
    cause = e.__cause__
    depth = 0
    while cause is not None and depth < 5:
        parts.append(f"  caused by -> {type(cause).__name__}: {cause}")
        cause = cause.__cause__
        depth += 1
    return "\n".join(parts)


def ask_claude():
    from anthropic import Anthropic

    def _call():
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"].strip(), timeout=30.0)
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=150,
                messages=[{"role": "user", "content": PROMPT}],
            )
        except Exception as e:
            print(f"  [Claude detailed error]\n{_describe_exception(e)}")
            raise
        return response.content[0].text

    return _retry_call(_call, "Claude")


def ask_chatgpt():
    from openai import OpenAI

    def _call():
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip(), timeout=30.0)
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": PROMPT}],
                max_completion_tokens=150,
            )
        except Exception as e:
            print(f"  [ChatGPT detailed error]\n{_describe_exception(e)}")
            raise
        return response.choices[0].message.content

    return _retry_call(_call, "ChatGPT")


def ask_gemini():
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip())

    models_to_try = [GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK]
    last_error = None

    for model_name in models_to_try:
        def _call():
            try:
                response = client.models.generate_content(model=model_name, contents=PROMPT)
            except Exception as e:
                print(f"  [Gemini ({model_name}) detailed error]\n{_describe_exception(e)}")
                raise
            return response.text
        try:
            return _retry_call(_call, f"Gemini ({model_name})")
        except Exception as e:
            last_error = e
            print(f"  Giving up on {model_name}, trying next option if available...")

    raise last_error


def parse_picks_with_confidence(raw_text, expected=5):
    """Extract (ticker, confidence) pairs. Handles class shares like
    BRK.B / BRK-B (normalized to BRK-B, the format yfinance uses).
    Mag 7 tickers are kept even though the prompt rules them out;
    they're only flagged in the run output."""
    if not raw_text:
        return []
    pairs = re.findall(r"\b([A-Z]{1,5}(?:[.\-][A-Z])?):\s*(\d{1,2})\b", raw_text)
    stopwords = {"I", "A", "THE", "AND", "OR", "US", "USD"}

    cleaned = []
    for t, c in pairs:
        t = t.replace(".", "-")
        c = int(c)
        if t in stopwords or not 1 <= c <= 10:
            continue
        if t in EXCLUDED_TICKERS:
            print(f"    Note: {t} breaks the no-Mag-7 rule, keeping it anyway")
        cleaned.append((t, c))
    return cleaned[:expected]


def get_todays_picks():
    picks = {}
    for source, fn in [("Claude", ask_claude), ("ChatGPT", ask_chatgpt), ("Gemini", ask_gemini)]:
        print(f"Asking {source}...")
        try:
            raw = fn()
            print(f"  Raw response: {raw}")
            picks[source] = parse_picks_with_confidence(raw)
        except Exception as e:
            print(f"  {source} failed after all retries: {e}")
            picks[source] = []
    print()
    for source, tickers in picks.items():
        print(f"{source} picks: {tickers}")
    print()
    return picks


# ---------------- Price fetching ----------------

def get_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        price = stock.fast_info.get("lastPrice")
        if price is None:
            hist = stock.history(period="1d")
            price = hist["Close"].iloc[-1] if not hist.empty else None
        return round(price, 2) if price is not None else None
    except Exception as e:
        print(f"  Could not fetch {ticker}: {e}")
        return None


# ---------------- Detailed log (per stock) ----------------

def _normalize_date_column(series):
    """Coerce a column of date-like strings (which may be a mix of
    formats, e.g. if the file was ever opened and saved in Excel) into
    a consistent YYYY-MM-DD string."""
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def _backfill_prompt_version(df):
    """Rows logged before v8 have no Prompt_Version; label them as the
    original open-ended prompt."""
    if "Prompt_Version" not in df.columns:
        df["Prompt_Version"] = None
    df["Prompt_Version"] = df["Prompt_Version"].fillna(LEGACY_PROMPT_VERSION)
    return df


def load_detail_log():
    if os.path.exists(DETAIL_FILE):
        df = pd.read_csv(DETAIL_FILE)
        for col in DETAIL_COLUMNS:
            if col not in df.columns:
                df[col] = None
        if not df.empty:
            df["Date_Picked"] = _normalize_date_column(df["Date_Picked"])
        return _backfill_prompt_version(df)
    return pd.DataFrame(columns=DETAIL_COLUMNS)


def fill_pending_day1(df, today_str):
    pending = df[(df["Price_Day1"].isna()) & (df["Date_Picked"] != today_str)]
    if pending.empty:
        print("No pending Day-1 prices to fill in.")
        return df

    print(f"Filling in Day-1 (next-day close) prices for {len(pending)} pick(s)...")
    for idx, row in pending.iterrows():
        price = get_price(row["Ticker"])
        df.at[idx, "Price_Day1"] = price
        print(f"  {row['Source']:8s} {row['Ticker']:6s} (picked {row['Date_Picked']}) -> Day1: {price}")
    return df


def log_todays_picks(df, today_str, today_picks):
    # Keyed on (Source, Ticker) so one AI's pick never blocks another's.
    if not df.empty:
        todays = df[df["Date_Picked"] == today_str]
        already_logged = set(zip(todays["Source"], todays["Ticker"]))
    else:
        already_logged = set()

    new_rows = []
    for source, pairs in today_picks.items():
        for ticker, confidence in pairs:
            ticker = ticker.strip().upper()
            if not ticker or (source, ticker) in already_logged:
                continue
            already_logged.add((source, ticker))
            price = get_price(ticker)
            print(f"  {source:8s} {ticker:6s} (confidence {confidence}) -> Day0: {price}")
            new_rows.append({
                "Date_Picked": today_str,
                "Source": source,
                "Ticker": ticker,
                "Confidence": confidence,
                "Price_Day0": price,
                "Price_Day1": None,
                "Prompt_Version": PROMPT_VERSION,
            })

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df


# ---------------- Portfolio value log (per AI, compounding) ----------------

def load_portfolio_log():
    if os.path.exists(PORTFOLIO_FILE):
        df = pd.read_csv(PORTFOLIO_FILE)
        if "Date" in df.columns and not df.empty:
            df["Date"] = _normalize_date_column(df["Date"])
        return _backfill_prompt_version(df)
    return pd.DataFrame(columns=PORTFOLIO_COLUMNS)


def update_portfolio_values(detail_df, portfolio_df):
    """
    For each (Date_Picked, Source) group in the detail log where every
    pick now has a Day1 price filled in, and that date+source isn't
    already recorded in the portfolio log, compute the average return
    across the picks and compound it into that AI's running portfolio
    value. Processes in chronological order per source.
    """
    already_done = set(zip(portfolio_df["Date"], portfolio_df["Source"])) if not portfolio_df.empty else set()

    new_rows = []
    for source in detail_df["Source"].dropna().unique():
        source_rows = detail_df[detail_df["Source"] == source]
        dates = sorted(source_rows["Date_Picked"].dropna().unique())

        source_portfolio_rows = portfolio_df[portfolio_df["Source"] == source] if not portfolio_df.empty else pd.DataFrame()
        if not source_portfolio_rows.empty:
            last_value = source_portfolio_rows.sort_values("Date")["Portfolio_Value"].iloc[-1]
        else:
            last_value = STARTING_PORTFOLIO_VALUE

        for date in dates:
            if (date, source) in already_done:
                continue
            day_picks = source_rows[source_rows["Date_Picked"] == date]
            if day_picks.empty or day_picks["Price_Day1"].isna().any() or day_picks["Price_Day0"].isna().any():
                continue

            returns = (day_picks["Price_Day1"] - day_picks["Price_Day0"]) / day_picks["Price_Day0"]
            avg_return = returns.mean()
            version = day_picks["Prompt_Version"].iloc[0]

            last_value = last_value * (1 + avg_return)
            new_rows.append({
                "Date": date,
                "Source": source,
                "Daily_Return_Pct": round(avg_return * 100, 3),
                "Portfolio_Value": round(last_value, 2),
                "Prompt_Version": version,
            })
            print(f"  Portfolio update: {source:8s} {date} [{version}] avg return {avg_return*100:+.2f}% "
                  f"-> ${last_value:,.2f}")

    if new_rows:
        portfolio_df = pd.concat([portfolio_df, pd.DataFrame(new_rows)], ignore_index=True)
    return portfolio_df


def main():
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"=== Running for {today_str} (prompt {PROMPT_VERSION}) ===\n")

    today_picks = get_todays_picks()

    detail_df = load_detail_log()

    print("Step 1: Checking for pending Day-1 prices from earlier picks...")
    detail_df = fill_pending_day1(detail_df, today_str)

    print("\nStep 2: Logging today's picks...")
    detail_df = log_todays_picks(detail_df, today_str, today_picks)

    detail_df = detail_df[DETAIL_COLUMNS]
    detail_df.to_csv(DETAIL_FILE, index=False)
    print(f"\nSaved detail log. {len(detail_df)} total rows in {DETAIL_FILE}")

    print("\nStep 3: Updating portfolio values...")
    portfolio_df = load_portfolio_log()
    portfolio_df = update_portfolio_values(detail_df, portfolio_df)
    portfolio_df = portfolio_df[PORTFOLIO_COLUMNS]
    portfolio_df.to_csv(PORTFOLIO_FILE, index=False)
    print(f"Saved portfolio log. {len(portfolio_df)} total rows in {PORTFOLIO_FILE}")


if __name__ == "__main__":
    main()