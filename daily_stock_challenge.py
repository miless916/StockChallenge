"""
AI Stock Picking Challenge - Full Automation (v5)
------------------------------------------------------------------------
Each time you run this script, it:
  1. Asks Claude, ChatGPT, and Gemini for 5 stock picks each, with a
     1-10 confidence rating for each pick (same prompt to all three).
     All three calls now retry automatically on transient connection
     errors (common on cloud CI runners like GitHub Actions).
  2. Parses each response into (ticker, confidence) pairs.
  3. Fills in "Day 1" (next trading day close) prices for any picks
     from a previous run that are still pending.
  4. Logs today's picks with today's closing price as "Day 0".
  5. Saves everything to stock_challenge_log.csv.

Run this once per day, ideally right around market close.

SETUP (one-time):
    pip install anthropic openai google-genai yfinance pandas openpyxl python-dotenv

    .env file in this folder with:
        ANTHROPIC_API_KEY=sk-ant-...
        OPENAI_API_KEY=sk-...
        GEMINI_API_KEY=AIza...

NOTE ON MODEL NAMES: check current model names if a call errors with
"model not found":
    - Anthropic: https://docs.claude.com/en/docs/about-claude/models
    - OpenAI:    https://platform.openai.com/docs/models
    - Gemini:    https://ai.google.dev/gemini-api/docs/models
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
GEMINI_MODEL_FALLBACK = "gemini-3.6-flash"  # tried if primary is overloaded

PROMPT = (
    "You are participating in a stock-picking challenge. Based on current "
    "market conditions, pick 5 publicly traded US stocks you believe are "
    "most likely to see their price increase over the next trading day. "
    "For each pick, also give a confidence rating from 1 (low) to 10 (high) "
    "reflecting how confident you are in that specific pick. "
    "Respond with ONLY a comma-separated list in this exact format, "
    "nothing else — no explanation, no extra text: "
    "TICKER:CONFIDENCE, TICKER:CONFIDENCE, TICKER:CONFIDENCE, TICKER:CONFIDENCE, TICKER:CONFIDENCE "
    "Example: AAPL:8, MSFT:6, GOOGL:7, AMZN:9, NVDA:5"
)

OUTPUT_FILE = "stock_challenge_log.csv"
COLUMNS = ["Date_Picked", "Source", "Ticker", "Confidence", "Price_Day0", "Price_Day1"]

TRANSIENT_ERROR_KEYWORDS = ["connection", "timeout", "timed out", "503", "unavailable", "overloaded"]


def _is_transient(error):
    text = str(error).lower()
    return any(keyword in text for keyword in TRANSIENT_ERROR_KEYWORDS)


def _retry_call(fn, label, max_retries=4):
    """Call fn() with retries on transient errors, exponential-ish backoff."""
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
                raise  # not transient — fail immediately, no point retrying
    raise last_error


# ---------------- AI query functions ----------------

def _ipv4_http_client(timeout=30.0):
    """Build an httpx client that forces IPv4. Works around a known issue
    where some cloud CI runners (including GitHub Actions) have broken or
    unreliable IPv6 routing, causing connection failures to hosts that
    offer IPv6 addresses even though general internet access works fine."""
    import httpx
    transport = httpx.HTTPTransport(local_address="0.0.0.0")
    return httpx.Client(transport=transport, timeout=timeout)


def ask_claude():
    from anthropic import Anthropic

    def _call():
        client = Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            http_client=_ipv4_http_client(),
        )
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": PROMPT}],
        )
        return response.content[0].text

    return _retry_call(_call, "Claude")


def ask_chatgpt():
    from openai import OpenAI

    def _call():
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            http_client=_ipv4_http_client(),
        )
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            max_completion_tokens=150,
        )
        return response.choices[0].message.content

    return _retry_call(_call, "ChatGPT")


def ask_gemini():
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    models_to_try = [GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK]
    last_error = None

    for model_name in models_to_try:
        def _call():
            response = client.models.generate_content(model=model_name, contents=PROMPT)
            return response.text
        try:
            return _retry_call(_call, f"Gemini ({model_name})")
        except Exception as e:
            last_error = e
            print(f"  Giving up on {model_name}, trying next option if available...")

    raise last_error


def parse_picks_with_confidence(raw_text, expected=5):
    """Extract (ticker, confidence) pairs like 'AAPL:8' from text."""
    if not raw_text:
        return []
    pairs = re.findall(r"\b([A-Z]{1,5}):(\d{1,2})\b", raw_text)
    stopwords = {"I", "A", "THE", "AND", "OR", "US", "USD"}
    cleaned = [(t, int(c)) for t, c in pairs if t not in stopwords and 1 <= int(c) <= 10]
    return cleaned[:expected]


def get_todays_picks():
    """Query all three AIs and return {source: [(ticker, confidence), ...]}."""
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


# ---------------- Price fetching + CSV logging ----------------

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


def load_log():
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df
    return pd.DataFrame(columns=COLUMNS)


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
    already_logged = set(
        df[df["Date_Picked"] == today_str]["Ticker"].tolist()
    ) if not df.empty else set()

    new_rows = []
    for source, pairs in today_picks.items():
        for ticker, confidence in pairs:
            ticker = ticker.strip().upper()
            if not ticker or ticker in already_logged:
                continue
            price = get_price(ticker)
            print(f"  {source:8s} {ticker:6s} (confidence {confidence}) -> Day0: {price}")
            new_rows.append({
                "Date_Picked": today_str,
                "Source": source,
                "Ticker": ticker,
                "Confidence": confidence,
                "Price_Day0": price,
                "Price_Day1": None,
            })

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df


def main():
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"=== Running for {today_str} ===\n")

    today_picks = get_todays_picks()

    df = load_log()

    print("Step 1: Checking for pending Day-1 prices from earlier picks...")
    df = fill_pending_day1(df, today_str)

    print("\nStep 2: Logging today's picks...")
    df = log_todays_picks(df, today_str, today_picks)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved. Log now has {len(df)} total rows in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()