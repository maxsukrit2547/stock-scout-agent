"""
Stock Scout Agent
Modes:
  scout  - news scan + drawdown check, every 4h weekdays
  deep   - structured pre-market brief, 18:30 BKK weekdays
  listen - inbound Telegram (trades, URLs, images, questions) + drawdown check, every 15 min
"""

import os
import re
import sys
import json
import html
import hashlib
import subprocess
import traceback
from datetime import datetime, timedelta

import requests
import feedparser
import yfinance as yf
import pandas as pd
import numpy as np

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai not installed. pip install google-genai", file=sys.stderr)
    raise


# ===================================================================
# CONFIG
# ===================================================================
DEFAULT_PORTFOLIO = {
    "AMZN":  {"shares": 3.13269,  "avg_cost": 223.28},
    "ASTS":  {"shares": 7.83626,  "avg_cost": 63.85},
    "AVGO":  {"shares": 1.45697,  "avg_cost": 343.19},
    "CRWD":  {"shares": 0.47362,  "avg_cost": 421.60},
    "GOOGL": {"shares": 2.19189,  "avg_cost": 228.25},
    "IONQ":  {"shares": 5.37155,  "avg_cost": 55.83},
    "META":  {"shares": 0.63592,  "avg_cost": 635.96},
    "MSFT":  {"shares": 1.15654,  "avg_cost": 432.61},
    "MU":    {"shares": 2.83080,  "avg_cost": 211.67},
    "NVDA":  {"shares": 4.06307,  "avg_cost": 147.78},
    "ORCL":  {"shares": 1.56845,  "avg_cost": 260.92},
    "RKLB":  {"shares": 16.84020, "avg_cost": 41.56},
    "TSLA":  {"shares": 0.28565,  "avg_cost": 349.63},
    "TSM":   {"shares": 1.87983,  "avg_cost": 265.50},
}

PORTFOLIO_FILE = "portfolio.json"
CACHE_FILE = "scout_cache.json"

MODEL_LIGHT = "gemini-2.5-flash-lite"
MODEL_DEEP  = "gemini-2.5-flash"

NORMAL_MULTIPLES = {"pe": 22, "ps": 6, "pb": 4}
TG_SAFE_LIMIT    = 3800
URL_RE           = re.compile(r'https?://\S+')
UNSUPPORTED_DOMAINS = ["tiktok.com", "facebook.com", "instagram.com", "fb.com"]

TARGET_PORTFOLIO_RISK_PCT = 1.0
MAX_POSITION_PCT          = 10.0
DRAWDOWN_WARN_PCT         = -8.0
DRAWDOWN_ALARM_PCT        = -15.0
DRAWDOWN_RESET_PCT        = -3.0

EARNINGS_HORIZON_DAYS  = 7
INSIDER_LOOKBACK_DAYS  = 90
INSIDER_MATERIAL_USD   = 500_000

WATCHLIST_BY_SECTOR = {
    "Semis":         ["AMD","INTC","MRVL","QCOM","ARM","ASML","AMAT","LRCX","KLAC","SMCI"],
    "Cloud_SaaS":    ["NOW","PANW","ZS","DDOG","SNOW","MDB","NET","OKTA","WDAY","ADBE"],
    "Internet":      ["NFLX","SPOT","RDDT","UBER","ABNB","BKNG"],
    "Space_Defense": ["LMT","RTX","BA","GD","PLTR","BKSY","PL","KTOS"],
    "Quantum_AI":    ["RGTI","QBTS","AI","BBAI","SOUN"],
    "EV_Auto":       ["RIVN","LCID","F","GM"],
    "Sector_ETFs":   ["XLK","SOXX","IGV","ARKX","XLC","XLE","XLF"],
}

RSS_FEEDS_BASE = [
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "https://www.cnbc.com/id/19854910/device/rss/rss.html",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://seekingalpha.com/market_currents.xml",
]


def _require_env(name):
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: missing required env var {name}", file=sys.stderr)
        sys.exit(1)
    return val


_client = None

def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=_require_env("GEMINI_API_KEY"))
    return _client


# ===================================================================
# PERSISTENCE
# ===================================================================
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"portfolio load err: {e}")
    return dict(DEFAULT_PORTFOLIO)


def save_portfolio(p):
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(p, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"portfolio save err: {e}")


def load_cache():
    try:
        with open(CACHE_FILE) as f:
            c = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        c = {}
    c.setdefault("seen_news", [])
    c.setdefault("fundamentals", {})
    c.setdefault("last_telegram_offset", 0)
    c.setdefault("portfolio_peak_value", 0)
    c.setdefault("peak_date", None)
    c.setdefault("last_drawdown_alert_level", None)
    return c


def save_cache(c):
    try:
        c["seen_news"] = c.get("seen_news", [])[-2000:]
        with open(CACHE_FILE, "w") as f:
            json.dump(c, f, default=str)
    except Exception as e:
        print(f"cache save err: {e}")


# ===================================================================
# DATA: news, fundamentals, earnings, insiders
# ===================================================================
def get_rss_feeds(portfolio_keys):
    feeds = list(RSS_FEEDS_BASE)
    for t in portfolio_keys:
        feeds.append(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US")
    return feeds


def fetch_news(portfolio_keys):
    items = []
    for url in get_rss_feeds(portfolio_keys):
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:20]:
                link  = getattr(e, "link",  "") or ""
                title = getattr(e, "title", "") or ""
                if not link or not title:
                    continue
                items.append({
                    "id":    hashlib.md5(link.encode()).hexdigest(),
                    "title": title,
                })
        except Exception as ex:
            print(f"feed err {url}: {ex}")
    return items


def fetch_ticker_news(ticker, limit=10):
    items = []
    try:
        feed = feedparser.parse(
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        )
        for e in feed.entries[:limit]:
            title = getattr(e, "title", "") or ""
            if title:
                items.append(f"- {title}")
    except Exception as ex:
        print(f"ticker news err {ticker}: {ex}")
    return "\n".join(items) if items else "(no recent news)"


def get_fundamentals(ticker, cache):
    key    = ticker.upper()
    cached = cache["fundamentals"].get(key)
    now    = datetime.utcnow().timestamp()
    if cached and (now - cached.get("fetched_at", 0)) < 86400:
        return cached
    try:
        info = yf.Ticker(ticker).info or {}
        data = {
            "ticker":    key,
            "price":     info.get("currentPrice") or info.get("regularMarketPrice"),
            "pe":        info.get("trailingPE"),
            "fwd_pe":    info.get("forwardPE"),
            "ps":        info.get("priceToSalesTrailing12Months"),
            "pb":        info.get("priceToBook"),
            "div_yield": info.get("dividendYield") or 0,
            "div_rate":  info.get("dividendRate")  or 0,
            "growth":    info.get("earningsGrowth"),
            "sector":    info.get("sector",   ""),
            "industry":  info.get("industry", ""),
            "market_cap":info.get("marketCap"),
            "fetched_at":now,
        }
        cache["fundamentals"][key] = data
        return data
    except Exception as e:
        print(f"yf fund err {ticker}: {e}")
        return cached


def get_next_earnings(ticker):
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None:
            return None, None
        date = None
        if isinstance(cal, dict):
            d = cal.get("Earnings Date")
            if isinstance(d, list) and d:
                date = d[0]
            elif d:
                date = d
        elif hasattr(cal, "iloc") and not cal.empty:
            try:
                date = cal.iloc[0].get("Earnings Date")
            except Exception:
                pass
        if date is None:
            return None, None
        if isinstance(date, str):
            date = pd.to_datetime(date)
        if hasattr(date, "to_pydatetime"):
            date = date.to_pydatetime()
        if hasattr(date, "tzinfo") and date.tzinfo:
            date = date.replace(tzinfo=None)
        days = (date - datetime.utcnow()).days
        return date.strftime("%b %d"), days
    except Exception as e:
        print(f"earnings err {ticker}: {e}")
        return None, None


def get_insider_activity(ticker, days=INSIDER_LOOKBACK_DAYS):
    try:
        trans = yf.Ticker(ticker).insider_transactions
        if trans is None or len(trans) == 0:
            return None
        df = trans.copy()
        date_col = next(
            (c for c in ["Start Date", "Date", "Latest Trans Date"] if c in df.columns), None
        )
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            cutoff = datetime.utcnow() - timedelta(days=days)
            df = df[df[date_col] >= cutoff]
        if len(df) == 0:
            return None
        trans_col = next((c for c in ["Transaction", "Text"] if c in df.columns), None)
        value_col = next((c for c in ["Value", "Position $ Value"] if c in df.columns), None)
        if not trans_col or not value_col:
            return None
        buys  = df[df[trans_col].astype(str).str.contains("Buy|Purchase|Acquisition", case=False, na=False)]
        sells = df[df[trans_col].astype(str).str.contains("Sale|Sell|Disposition",    case=False, na=False)]
        buy_value  = float(buys[value_col].sum())  if len(buys)  else 0.0
        sell_value = float(sells[value_col].sum()) if len(sells) else 0.0
        if max(abs(buy_value), abs(sell_value)) < INSIDER_MATERIAL_USD:
            return None
        return {
            "buy_count":  len(buys),
            "sell_count": len(sells),
            "buy_value":  buy_value,
            "sell_value": sell_value,
            "net_value":  buy_value - sell_value,
            "days":       days,
        }
    except Exception as e:
        print(f"insider err {ticker}: {e}")
        return None


# ===================================================================
# ANALYSIS: valuation, technicals, volatility, sizing
# ===================================================================
def pick_valuation(fund):
    if not fund or not fund.get("price"):
        return None
    price = fund["price"]
    pe, ps, pb  = fund.get("pe"), fund.get("ps"), fund.get("pb")
    div_yield   = fund.get("div_yield") or 0
    div_rate    = fund.get("div_rate")  or 0
    fair, method = None, None
    if div_yield >= 0.02 and div_rate > 0:
        fair   = round((div_rate * 1.05) / 0.04, 2)
        method = "DDM"
    elif pe and pe > 0 and pe < 200:
        fair   = round(price * NORMAL_MULTIPLES["pe"] / pe, 2)
        method = "P/E"
    elif ps and ps > 0:
        fair   = round(price * NORMAL_MULTIPLES["ps"] / ps, 2)
        method = "P/S"
    elif pb and pb > 0:
        fair   = round(price * NORMAL_MULTIPLES["pb"] / pb, 2)
        method = "P/BV"
    if not fair:
        return None
    upside = round((fair - price) / price * 100, 1)
    verdict = "undervalued" if upside > 15 else ("overvalued" if upside < -15 else "fair")
    return {
        "fair_value":    fair,
        "method":        method,
        "upside_pct":    upside,
        "verdict":       verdict,
        "current_price": round(price, 2),
    }


def technical_signals(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="3mo", interval="1d")
        if hist is None or hist.empty or len(hist) < 50:
            return None
        close = hist["Close"]
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi_series = 100 - 100 / (1 + rs)
        rsi_now = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
        ema12     = close.ewm(span=12, adjust=False).mean()
        ema26     = close.ewm(span=26, adjust=False).mean()
        hist_macd = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
        ma50      = close.rolling(50).mean()
        above_ma50 = bool(close.iloc[-1] > ma50.iloc[-1])
        macd_bull  = hist_macd.iloc[-1] > 0 and hist_macd.iloc[-1] > hist_macd.iloc[-2]
        macd_bear  = hist_macd.iloc[-1] < 0 and hist_macd.iloc[-1] < hist_macd.iloc[-2]
        return {
            "rsi":        rsi_now,
            "macd_bull":  bool(macd_bull),
            "macd_bear":  bool(macd_bear),
            "above_ma50": above_ma50,
            "support":    round(float(close.tail(20).min()), 2),
            "resistance": round(float(close.tail(20).max()), 2),
        }
    except Exception as e:
        print(f"ta err {ticker}: {e}")
        return None


def tech_verdict(tech):
    if not tech:
        return "no technical data"
    rsi          = tech["rsi"]
    above, bull, bear = tech["above_ma50"], tech["macd_bull"], tech["macd_bear"]
    if rsi >= 75 and bear:        return "extremely overbought, reversal forming"
    if rsi >= 70:                 return "overbought territory"
    if rsi <= 25 and bull:        return "extremely oversold, reversal setup"
    if rsi <= 30 and bull and above: return "oversold dip-buy setup"
    if rsi <= 30:                 return "oversold, awaiting reversal"
    if above and bull:            return "healthy uptrend with momentum"
    if above and not bull:        return "uptrend, momentum cooling"
    if not above and bull:        return "recovering from weakness"
    return "weak trend"


def compute_volatility(ticker, days=30):
    try:
        hist = yf.Ticker(ticker).history(period="3mo", interval="1d")
        if hist is None or hist.empty:
            return None
        returns = hist["Close"].pct_change().dropna()
        if len(returns) < days:
            return None
        return float(returns.tail(days).std() * np.sqrt(252))
    except Exception as e:
        print(f"vol err {ticker}: {e}")
        return None


def suggest_position_size(ticker, current_price, portfolio_value):
    vol = compute_volatility(ticker)
    if not vol or not current_price or not portfolio_value:
        return None
    daily_vol = vol / np.sqrt(252)
    if daily_vol <= 0:
        return None
    target_dollar_risk = portfolio_value * (TARGET_PORTFOLIO_RISK_PCT / 100)
    max_position       = portfolio_value * (MAX_POSITION_PCT / 100)
    suggested_dollar   = min(target_dollar_risk / daily_vol, max_position)
    return {
        "total_dollars":     round(suggested_dollar, 0),
        "total_shares":      round(suggested_dollar / current_price, 4),
        "as_pct_portfolio":  round(suggested_dollar / portfolio_value * 100, 1),
        "vol_annualized_pct":round(vol * 100, 1),
        "tranche_1":         round(suggested_dollar * 0.20, 0),
        "tranche_2":         round(suggested_dollar * 0.30, 0),
        "tranche_3":         round(suggested_dollar * 0.50, 0),
    }


# ===================================================================
# PORTFOLIO VALUATION + DRAWDOWN
# ===================================================================
def compute_portfolio_value(portfolio, cache):
    total = 0.0
    for ticker, d in portfolio.items():
        fund = get_fundamentals(ticker, cache)
        if fund and fund.get("price"):
            total += fund["price"] * d["shares"]
    return total


def check_drawdown_alert(portfolio_value, cache):
    peak = cache.get("portfolio_peak_value", 0) or 0
    if portfolio_value > peak:
        cache["portfolio_peak_value"]     = portfolio_value
        cache["peak_date"]                = datetime.utcnow().strftime("%b %d")
        cache["last_drawdown_alert_level"] = None
        return None
    if peak <= 0:
        return None
    dd_pct     = (portfolio_value - peak) / peak * 100
    last_alert = cache.get("last_drawdown_alert_level")
    if dd_pct > DRAWDOWN_RESET_PCT and last_alert is not None:
        cache["last_drawdown_alert_level"] = None
        return None
    if dd_pct <= DRAWDOWN_ALARM_PCT and last_alert != "alarm":
        cache["last_drawdown_alert_level"] = "alarm"
        return {"level": "alarm", "dd_pct": dd_pct, "current": portfolio_value,
                "peak": peak, "peak_date": cache.get("peak_date", "—")}
    if dd_pct <= DRAWDOWN_WARN_PCT and last_alert is None:
        cache["last_drawdown_alert_level"] = "warning"
        return {"level": "warning", "dd_pct": dd_pct, "current": portfolio_value,
                "peak": peak, "peak_date": cache.get("peak_date", "—")}
    return None


def send_drawdown_alert(alert):
    emoji, label = ("🚨", "DRAWDOWN ALARM") if alert["level"] == "alarm" else ("⚠️", "DRAWDOWN WARNING")
    msg = (
        f"{emoji} {bold(label)}\n\n"
        f"Portfolio: {code('$' + str(round(alert['current'])))} "
        f"({code('{:.1f}%'.format(alert['dd_pct']))} from peak)\n"
        f"Peak: {code('$' + str(round(alert['peak'])))} on {esc(alert['peak_date'])}\n\n"
        f"<i>Review thesis on biggest losers before adding more. "
        f"Don't deploy fresh capital into falling positions until fundamentals are confirmed intact.</i>"
    )
    telegram_send(msg)


# ===================================================================
# GEMINI
# ===================================================================
def call_gemini(prompt, max_tokens=600, json_mode=False, model=None):
    try:
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=0.3,
            response_mime_type="application/json" if json_mode else None,
        )
        response = get_client().models.generate_content(
            model=model or MODEL_LIGHT, contents=prompt, config=config,
        )
        return response.text or ("{}" if json_mode else "")
    except Exception as e:
        print(f"gemini err: {e}")
        return "{}" if json_mode else ""


def call_gemini_image(image_bytes, prompt, mime_type="image/jpeg", max_tokens=1500):
    try:
        config = types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=0.4)
        response = get_client().models.generate_content(
            model=MODEL_DEEP,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt],
            config=config,
        )
        return response.text or ""
    except Exception as e:
        print(f"gemini image err: {e}")
        return ""


# ===================================================================
# TELEGRAM: helpers + send + chunk
# ===================================================================
def esc(s):
    if s is None:
        return ""
    return html.escape(str(s), quote=False)

def code(s):
    return f"<code>{esc(s)}</code>"

def bold(s):
    return f"<b>{esc(s)}</b>"


def _split_at_lines(text, limit):
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit:
            if cur:
                chunks.append(cur)
                cur = line
            else:
                chunks.append(line[:limit])
                cur = line[limit:]
        else:
            cur = cur + "\n" + line if cur else line
    if cur:
        chunks.append(cur)
    return chunks


def chunk_sections(sections, limit=TG_SAFE_LIMIT):
    messages, current = [], ""
    for sec in sections:
        if not sec:
            continue
        if len(sec) > limit:
            if current:
                messages.append(current)
                current = ""
            for sub in _split_at_lines(sec, limit):
                messages.append(sub)
            continue
        sep = "\n\n" if current else ""
        if len(current) + len(sep) + len(sec) <= limit:
            current = current + sep + sec
        else:
            messages.append(current)
            current = sec
    if current:
        messages.append(current)
    return messages


def telegram_send(text, html_mode=True):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat  = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("WARN: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing; skipping send")
        return
    payload = {"chat_id": chat, "text": text, "disable_web_page_preview": True}
    if html_mode:
        payload["parse_mode"] = "HTML"
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload, timeout=15,
        )
        if r.status_code != 200 and html_mode:
            print(f"tg HTML failed {r.status_code}: {r.text[:200]}")
            payload.pop("parse_mode", None)
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data=payload, timeout=15,
            )
    except Exception as e:
        print(f"tg send err: {e}")


def send_chunked(sections):
    for msg in chunk_sections(sections):
        telegram_send(msg, html_mode=True)


# ===================================================================
# TELEGRAM: receive
# ===================================================================
def telegram_get_updates(offset=None):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return []
    params = {"offset": offset} if offset else {}
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{token}/getUpdates",
            params=params, timeout=10,
        )
        return r.json().get("result", [])
    except Exception as e:
        print(f"tg get_updates err: {e}")
        return []


def download_telegram_file(file_id):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return None
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{token}/getFile",
            params={"file_id": file_id}, timeout=10,
        )
        file_path = (r.json().get("result") or {}).get("file_path")
        if not file_path:
            return None
        r = requests.get(
            f"https://api.telegram.org/file/bot{token}/{file_path}", timeout=20,
        )
        return r.content if r.status_code == 200 else None
    except Exception as e:
        print(f"tg download err: {e}")
        return None


# ===================================================================
# INTERACTIVE: image / URL / question handlers
# ===================================================================
def fetch_url_text(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        text = r.text
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>',  '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&nbsp;|&amp;|&#\d+;', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:6000]
    except Exception as e:
        print(f"url fetch err: {e}")
        return None


def is_unsupported_url(url):
    return any(d in url.lower() for d in UNSUPPORTED_DOMAINS)


def portfolio_summary_str(portfolio):
    return ", ".join(f"{t}({d['shares']:.2f}@${d['avg_cost']})" for t, d in portfolio.items())


def analyze_image(image_bytes, caption, portfolio):
    user_q = caption or "Analyze this image and tell me how it might affect my portfolio."
    prompt = (
        f"User's portfolio: {portfolio_summary_str(portfolio)}\n\n"
        f'User sent an image with: "{user_q}"\n\n'
        "The image could be a chart, news headline, social media post, or research screenshot. Analyze:\n"
        "1. What the image shows (1 sentence)\n"
        "2. Direct impact on portfolio holdings (which tickers, how)\n"
        "3. Sector/macro implications\n"
        "4. Actionable takeaway\n\n"
        "Use HTML for Telegram: <b>bold</b>, <i>italic</i>, <code>monospace</code>. Under 1200 chars."
    )
    try:
        return call_gemini_image(image_bytes, prompt)
    except Exception as e:
        print(f"image analyze err: {e}")
        return None


def analyze_url(url, portfolio):
    if is_unsupported_url(url):
        return ("That platform (TikTok/Facebook/Instagram) blocks direct fetching. "
                "Send me a <b>screenshot</b> of the post instead.")
    content = fetch_url_text(url)
    if not content or len(content) < 200:
        return "Couldn't access that URL or content was too thin. Try a screenshot or paste the text."
    prompt = (
        f"User's portfolio: {portfolio_summary_str(portfolio)}\n\n"
        f"User shared URL: {url}\n\nPage content:\n{content}\n\n"
        "Respond:\n1. What this is about (1 sentence)\n2. Impact on portfolio holdings\n"
        "3. Indirect/sector impact\n4. Actionable takeaway\n\n"
        "Use HTML for Telegram. Under 1200 chars."
    )
    return call_gemini(prompt, max_tokens=1500, model=MODEL_DEEP)


def analyze_question(text, portfolio):
    prompt = (
        f"User's portfolio: {portfolio_summary_str(portfolio)}\n\n"
        f'User asked: "{text}"\n\n'
        "Answer with portfolio context. If about a stock or news, analyze impact on holdings. "
        "If general market question, give focused take. Use HTML for Telegram. Under 1000 chars."
    )
    return call_gemini(prompt, max_tokens=1500, model=MODEL_DEEP)


# ===================================================================
# TRADE PARSING
# ===================================================================
def parse_trade_command(message_text, current_tickers):
    prompt = (
        "Parse this user message about a stock trade into JSON.\n\n"
        f"Current portfolio: {', '.join(current_tickers)}\n"
        f'Message: "{message_text}"\n\n'
        "Actions: buy, sell, set, remove, ignore.\n"
        'Return JSON: {"action":"...","ticker":"MU","shares":1.5,'
        '"is_percentage":false,"price":750.5,"new_avg_cost":null,'
        '"new_shares":null,"confidence":0.95,"summary":"..."}\n\n'
        "Rules: 'sold 30%'->shares=30,is_percentage=true. 'bought N at P'->shares=N,price=P. "
        "'sold all X'->shares=100,is_percentage=true. If unclear, action=ignore. Confidence<0.7 skipped."
    )
    raw = call_gemini(prompt, max_tokens=400, json_mode=True)
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"trade parse err: {e}")
        return {"action": "ignore", "confidence": 0}


def apply_trade(portfolio, trade):
    if trade.get("action") == "ignore" or trade.get("confidence", 0) < 0.7:
        return portfolio, None
    ticker = (trade.get("ticker") or "").upper()
    if not ticker:
        return portfolio, None
    p = {k: dict(v) for k, v in portfolio.items()}
    a = trade["action"]

    if a == "sell":
        if ticker not in p:
            return portfolio, f"⚠️ Cannot sell {ticker} — not in portfolio"
        if trade.get("is_percentage"):
            pct = (trade.get("shares") or 0) / 100
            if pct >= 1:
                old = p.pop(ticker)
                return p, f"✅ Sold all of {ticker} (was {old['shares']:.4f} sh @ ${old['avg_cost']})"
            old_sh = p[ticker]["shares"]
            new_sh = round(old_sh * (1 - pct), 6)
            p[ticker]["shares"] = new_sh
            return p, f"✅ Sold {pct*100:.0f}% of {ticker}: {old_sh:.4f} → {new_sh:.4f} sh"
        sold  = trade.get("shares") or 0
        old_sh = p[ticker]["shares"]
        new_sh = round(old_sh - sold, 6)
        if new_sh <= 0:
            p.pop(ticker)
            return p, f"✅ Sold {sold} sh {ticker} (closed)"
        p[ticker]["shares"] = new_sh
        return p, f"✅ Sold {sold} {ticker}: {old_sh:.4f} → {new_sh:.4f}"

    if a == "buy":
        new_sh = trade.get("shares") or 0
        price  = trade.get("price")
        if not new_sh or not price:
            return portfolio, "⚠️ Buy needs share count and price"
        if ticker in p:
            old_sh   = p[ticker]["shares"]
            old_cost = p[ticker]["avg_cost"]
            total    = old_sh + new_sh
            new_avg  = (old_sh * old_cost + new_sh * price) / total
            p[ticker] = {"shares": round(total, 6), "avg_cost": round(new_avg, 2)}
            return p, f"✅ Bought {new_sh} {ticker} @ ${price}. Now {total:.4f} sh @ avg ${new_avg:.2f}"
        p[ticker] = {"shares": round(new_sh, 6), "avg_cost": round(price, 2)}
        return p, f"✅ New position: {new_sh} {ticker} @ ${price}"

    if a == "set":
        ns, nc = trade.get("new_shares"), trade.get("new_avg_cost")
        if ns is None or nc is None:
            return portfolio, "⚠️ Set needs both shares and avg_cost"
        p[ticker] = {"shares": round(ns, 6), "avg_cost": round(nc, 2)}
        return p, f"✅ Set {ticker} to {ns} sh @ avg ${nc}"

    if a == "remove":
        if ticker in p:
            p.pop(ticker)
            return p, f"✅ Removed {ticker}"
        return portfolio, f"⚠️ {ticker} not in portfolio"

    return portfolio, None


def git_commit_portfolio():
    try:
        if not os.path.isdir(".git"):
            return
        subprocess.run(["git", "config", "user.name",  "scout-agent"],   check=False)
        subprocess.run(["git", "config", "user.email", "scout@bot.local"], check=False)
        subprocess.run(["git", "add", PORTFOLIO_FILE], check=False)
        if subprocess.run(["git", "diff", "--staged", "--quiet"]).returncode == 0:
            return
        subprocess.run(["git", "commit", "-m", "Update portfolio from Telegram"], check=False)
        subprocess.run(["git", "push"], check=False)
    except Exception as e:
        print(f"git commit err: {e}")


# ===================================================================
# UNIFIED MESSAGE PROCESSOR
# ===================================================================
def process_telegram_messages(cache, portfolio):
    last_offset = cache.get("last_telegram_offset", 0)
    updates = telegram_get_updates(offset=last_offset + 1 if last_offset else None)
    if not updates:
        return portfolio

    p = portfolio
    for u in updates:
        try:
            cache["last_telegram_offset"] = u["update_id"]
            msg     = u.get("message") or {}
            text    = (msg.get("text")    or "").strip()
            caption = (msg.get("caption") or "").strip()
            photos  = msg.get("photo")
            full_text = text or caption

            if full_text.startswith("/"):
                continue

            if photos:
                largest = max(photos, key=lambda x: x.get("file_size", 0) or 0)
                img = download_telegram_file(largest["file_id"])
                if img:
                    resp = analyze_image(img, caption, p)
                    telegram_send(
                        f"🖼 <b>Image Analysis</b>\n\n{resp}" if resp
                        else "⚠️ Couldn't analyze that image."
                    )
                continue

            if not text:
                continue

            url_match = URL_RE.search(text)
            if url_match:
                url  = url_match.group(0).rstrip(".,;)")
                resp = analyze_url(url, p)
                if resp:
                    telegram_send(f"🔗 <b>Link Analysis</b>\n\n{resp}")
                continue

            trade = parse_trade_command(text, list(p.keys()))
            if trade.get("confidence", 0) >= 0.7 and trade.get("action") not in (None, "ignore"):
                new_p, conf = apply_trade(p, trade)
                if conf:
                    if not conf.startswith("⚠️"):
                        p = new_p
                        save_portfolio(p)
                        git_commit_portfolio()
                    telegram_send(f"📒 <b>Portfolio Update</b>\n\n{esc(conf)}")
                    continue

            resp = analyze_question(text, p)
            if resp:
                telegram_send(f"💬 <b>Analysis</b>\n\n{resp}")

        except Exception as e:
            print(f"process update err: {e}")
            traceback.print_exc()

    return p


# ===================================================================
# DEEP RESEARCH
# ===================================================================
def deep_research_data(cache, portfolio):
    sections = []
    earnings_this_week = []

    for ticker, d in portfolio.items():
        fund = get_fundamentals(ticker, cache)
        if not fund or not fund.get("price"):
            continue
        val     = pick_valuation(fund)
        tech    = technical_signals(ticker)
        verdict = tech_verdict(tech)
        pl_pct  = (fund["price"] - d["avg_cost"]) / d["avg_cost"] * 100
        news    = fetch_ticker_news(ticker, limit=8)

        earn_date, earn_days = get_next_earnings(ticker)
        insider = get_insider_activity(ticker)

        if earn_date and earn_days is not None and 0 <= earn_days <= EARNINGS_HORIZON_DAYS:
            earnings_this_week.append({"ticker": ticker, "date": earn_date, "days": earn_days})

        block  = f"## {ticker} ({fund.get('industry') or fund.get('sector')})\n"
        block += (f"Position: {d['shares']:.4f} sh @ ${d['avg_cost']} "
                  f"| Now ${fund['price']:.2f} | P&L {pl_pct:+.1f}%\n")
        if val:
            block += (f"Fair value: ${val['fair_value']} via {val['method']} "
                      f"({val['upside_pct']:+.1f}%, {val['verdict']})\n")
        block += f"Technical: {verdict}\n"
        if tech:
            block += f"Range: ${tech['support']}-${tech['resistance']}\n"
        if earn_date:
            block += f"Next earnings: {earn_date} ({earn_days} days)\n"
        if insider:
            net = insider["net_value"]
            if net > 0:
                block += (f"Insiders ({INSIDER_LOOKBACK_DAYS}d): "
                          f"+${net/1e6:.2f}M net buys "
                          f"({insider['buy_count']} buyers, {insider['sell_count']} sellers)\n")
            else:
                block += (f"Insiders ({INSIDER_LOOKBACK_DAYS}d): "
                          f"-${abs(net)/1e6:.2f}M net sales "
                          f"({insider['buy_count']} buyers, {insider['sell_count']} sellers)\n")
        block += f"News:\n{news}"
        sections.append(block)

    full_context = "\n\n".join(sections)
    earnings_str = (
        ", ".join(f"{e['ticker']} {e['date']} ({e['days']}d)" for e in earnings_this_week)
        or "none"
    )

    prompt = (
        "You are a portfolio analyst. Below is the user's portfolio with valuation verdicts, "
        "technical state in plain English, recent news, next earnings, and insider activity. "
        "Return STRICT JSON only.\n\n"
        f"Earnings within {EARNINGS_HORIZON_DAYS} days: {earnings_str}\n\n"
        f"{full_context}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "executive_summary": {"overall_status":"...","risk":"...","opportunity":"..."},\n'
        '  "earnings_this_week": [{"ticker":"AVGO","date":"May 14","days":5,"watch_for":"AI revenue commentary"}],\n'
        '  "action_priorities": {\n'
        '    "trim_take_profits": [{"ticker":"MU","action":"...","trigger":"..."}],\n'
        '    "monitor_for_exit":  [{"ticker":"ORCL","concern":"...","watch_for":"..."}]\n'
        "  },\n"
        '  "watch_at_open": [{"ticker":"MU","zone":"$745-755","verdict":"...","scenario":"..."}],\n'
        '  "macro_risks": ["...","...","..."],\n'
        '  "per_stock_status": {\n'
        '    "green":  [{"ticker":"RKLB","state":"...","news":"...","insider":"...","action":"..."}],\n'
        '    "yellow": [],\n'
        '    "red":    []\n'
        "  },\n"
        '  "sector_trends": [{"sector":"...","trend":"...","your_exposure":["..."],"outlook":"..."}],\n'
        '  "watchlist_opportunities": [{"ticker":"AMD","thesis":"...","why_now":"...","fair_value_estimate":220}]\n'
        "}\n\n"
        "Rules:\n"
        "- DO NOT include raw indicator numbers (RSI, P/E, etc.). Only verdicts.\n"
        "- per_stock_status: every portfolio ticker categorized exactly once.\n"
        "- earnings_this_week: only tickers with earnings <=7 days.\n"
        "- watch_at_open: exactly 3. macro_risks: exactly 3. action_priorities: max 5 combined.\n"
        "- sector_trends: 2-4 megatrends. watchlist_opportunities: 2-3 non-portfolio names.\n"
        "- All text concise for mobile."
    )

    raw = call_gemini(prompt, max_tokens=10000, model=MODEL_DEEP, json_mode=True)
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"deep parse err: {e}")
        return {
            "executive_summary":  {"overall_status": "Parse error", "risk": "—", "opportunity": "—"},
            "earnings_this_week": [],
            "action_priorities":  {"trim_take_profits": [], "monitor_for_exit": []},
            "watch_at_open":      [],
            "macro_risks":        [],
            "per_stock_status":   {"green": [], "yellow": [], "red": []},
            "sector_trends":      [],
            "watchlist_opportunities": [],
        }


# ===================================================================
# MESSAGE BUILDERS
# ===================================================================
def _market_status():
    """
    Returns a dynamic string based on real NYSE market state:
      Before open  → 'Xh Ym to US open'
      Market open  → 'Xh Ym to US close 🔔'
      Weekend gap  → '1d Xh Ym to US open'
    Handles EDT (UTC-4) and EST (UTC-5) automatically — no extra libraries needed.
    """
    now_utc = datetime.utcnow()
    year    = now_utc.year

    # ── DST: 2nd Sunday of March → 1st Sunday of November ──────────
    d = datetime(year, 3, 1)
    sundays = 0
    while sundays < 2:
        if d.weekday() == 6:
            sundays += 1
        if sundays < 2:
            d += timedelta(days=1)
    dst_start_utc = d.replace(hour=7, minute=0, second=0, microsecond=0)

    d = datetime(year, 11, 1)
    while d.weekday() != 6:
        d += timedelta(days=1)
    dst_end_utc = d.replace(hour=6, minute=0, second=0, microsecond=0)

    is_edt    = dst_start_utc <= now_utc < dst_end_utc
    et_offset = timedelta(hours=-4 if is_edt else -5)
    now_et    = now_utc + et_offset
    # ────────────────────────────────────────────────────────────────

    today_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    today_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
    is_weekday  = now_et.weekday() < 5

    # Market is currently OPEN
    if is_weekday and today_open <= now_et < today_close:
        secs  = int((today_close - now_et).total_seconds())
        hours = secs // 3600
        mins  = (secs % 3600) // 60
        if hours >= 1:
            return f"{hours}h {mins}m to US close 🔔"
        return f"{mins}m to US close 🔔"

    # Market is CLOSED — find next open
    candidate = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    for _ in range(7):
        if candidate.weekday() < 5 and candidate > now_et:
            break
        candidate += timedelta(days=1)
        candidate = candidate.replace(hour=9, minute=30, second=0, microsecond=0)

    secs  = int((candidate - now_et).total_seconds())
    days  = secs // 86400
    hours = (secs % 86400) // 3600
    mins  = (secs % 3600) // 60

    if days >= 1:
        return f"{days}d {hours}h {mins}m to US open"
    if hours >= 1:
        return f"{hours}h {mins}m to US open"
    return f"{mins}m to US open"


def build_header():
    bkk        = datetime.utcnow() + timedelta(hours=7)
    market_str = _market_status()
    return (
        f"📊 {bold('PRE-MARKET DEEP RESEARCH')}\n"
        f"<i>{esc(bkk.strftime('%b %d, %Y'))} · "
        f"{esc(bkk.strftime('%H:%M'))} BKK · "
        f"{esc(market_str)}</i>"
    )


def build_exec_summary(d):
    return (
        f"📑 {bold('EXECUTIVE SUMMARY')}\n\n"
        f"{bold('Status:')} {esc(d.get('overall_status', '—'))}\n\n"
        f"{bold('Top Risk:')} {esc(d.get('risk', '—'))}\n\n"
        f"{bold('Opportunity:')} {esc(d.get('opportunity', '—'))}"
    )


def build_earnings_this_week(items):
    parts = [f"📅 {bold('EARNINGS THIS WEEK')}"]
    if not items:
        parts.append("\n<i>No portfolio earnings in next 7 days.</i>")
        return "\n".join(parts)
    for it in items:
        line = f"\n• {code('$' + it['ticker'])} — {esc(it.get('date', '?'))}"
        if it.get("days") is not None:
            line += f" <i>(in {it['days']}d)</i>"
        if it.get("watch_for"):
            line += f"\n   ▸ {esc(it['watch_for'])}"
        parts.append(line)
    return "\n".join(parts)


def build_action_priorities(d):
    parts = [f"🎯 {bold('ACTION PRIORITIES')}"]
    trim  = d.get("trim_take_profits") or []
    if trim:
        parts.append(f"\n💰 {bold('Trim / Take Profits')}")
        for it in trim:
            line = f"• {code('$' + it['ticker'])} — {esc(it.get('action', ''))}"
            if it.get("trigger"):
                line += f"\n   <i>{esc(it['trigger'])}</i>"
            parts.append(line)
    exit_w = d.get("monitor_for_exit") or []
    if exit_w:
        parts.append(f"\n🚪 {bold('Monitor for Exit')}")
        for it in exit_w:
            line = f"• {code('$' + it['ticker'])} — {esc(it.get('concern', ''))}"
            if it.get("watch_for"):
                line += f"\n   <i>Watch: {esc(it['watch_for'])}</i>"
            parts.append(line)
    if not trim and not exit_w:
        parts.append("\n<i>No high-priority actions today.</i>")
    return "\n".join(parts)


def build_watch_at_open(items):
    parts = [f"👁️ {bold('WATCH AT OPEN')}"]
    if not items:
        parts.append("\n<i>No specific tickers flagged.</i>")
        return "\n".join(parts)
    for it in items[:3]:
        line = f"\n• {code('$' + it['ticker'])}"
        if it.get("zone"):
            line += f"   Zone: {code(it['zone'])}"
        if it.get("verdict"):
            line += f"\n   State: <i>{esc(it['verdict'])}</i>"
        if it.get("scenario"):
            line += f"\n   {esc(it['scenario'])}"
        parts.append(line)
    return "\n".join(parts)


def build_macro_risks(risks):
    parts = [f"⚠️ {bold('TOP 3 MACRO RISKS')}"]
    if not risks:
        parts.append("\n<i>No material macro risks.</i>")
    for i, r in enumerate(risks[:3], 1):
        parts.append(f"\n{i}. {esc(r)}")
    return "\n".join(parts)


def build_per_stock(d):
    parts = [f"📋 {bold('PER-STOCK CATALYSTS & STATUS')}"]

    def add_group(label, emoji, items):
        if not items:
            return
        parts.append(f"\n{emoji} {bold(label)}")
        for it in items:
            line = f"• {code('$' + it['ticker'])}: {esc(it.get('state', ''))}"
            if it.get("news"):
                line += f"\n   📰 <i>{esc(it['news'])}</i>"
            if it.get("insider"):
                line += f"\n   🤝 <i>{esc(it['insider'])}</i>"
            if it.get("action"):
                line += f"\n   ▸ {esc(it['action'])}"
            parts.append(line)

    add_group("Green",  "🟢", d.get("green")  or [])
    add_group("Yellow", "🟡", d.get("yellow") or [])
    add_group("Red",    "🔴", d.get("red")    or [])
    return "\n".join(parts)


def build_sector_trends(trends):
    parts = [f"🌐 {bold('SECTOR & MEGATRENDS')}"]
    if not trends:
        parts.append("\n<i>No notable sector moves.</i>")
        return "\n".join(parts)
    for t in trends:
        line = f"\n• {bold(t.get('sector', '—'))}"
        if t.get("outlook"):
            line += f" — <i>{esc(t['outlook'])}</i>"
        line += f"\n   {esc(t.get('trend', ''))}"
        if t.get("your_exposure"):
            tickers = " ".join(code("$" + x) for x in t["your_exposure"])
            line += f"\n   Your exposure: {tickers}"
        parts.append(line)
    return "\n".join(parts)


def enrich_watchlist_opps(opps, cache):
    for o in opps:
        ticker = (o.get("ticker") or "").upper()
        if not ticker:
            continue
        fund = get_fundamentals(ticker, cache)
        if fund and fund.get("price"):
            o["current_price"] = round(fund["price"], 2)
            fv = o.get("fair_value_estimate")
            if fv and fund["price"]:
                o["upside_pct"] = round((fv - fund["price"]) / fund["price"] * 100, 1)
    return opps


def build_watchlist_opps(opps):
    parts = [f"🔭 {bold('OPPORTUNITIES OUTSIDE PORTFOLIO')}"]
    if not opps:
        parts.append("\n<i>No clear setups outside your portfolio today.</i>")
        return "\n".join(parts)
    for o in opps[:3]:
        line = f"\n• {code('$' + (o.get('ticker') or ''))} — {esc(o.get('thesis', ''))}"
        if o.get("why_now"):
            line += f"\n   <i>Why now: {esc(o['why_now'])}</i>"
        cur = o.get("current_price")
        fv  = o.get("fair_value_estimate")
        if cur and fv:
            line += f"\n   Current: {code('$' + str(cur))} · Fair value: {code('$' + str(fv))}"
            up = o.get("upside_pct")
            if up is not None:
                line += f" ({code('{:+.1f}%'.format(up))})"
        elif cur:
            line += f"\n   Current: {code('$' + str(cur))}"
        elif fv:
            line += f"\n   Fair value est: {code('$' + str(fv))}"
        parts.append(line)
    return "\n".join(parts)


def build_brief_sections(brief_data):
    return [
        build_header(),
        build_exec_summary(brief_data.get("executive_summary") or {}),
        build_earnings_this_week(brief_data.get("earnings_this_week") or []),
        build_action_priorities(brief_data.get("action_priorities") or {}),
        build_watch_at_open(brief_data.get("watch_at_open") or []),
        build_macro_risks(brief_data.get("macro_risks") or []),
        build_per_stock(brief_data.get("per_stock_status") or {}),
        build_sector_trends(brief_data.get("sector_trends") or []),
        build_watchlist_opps(brief_data.get("watchlist_opportunities") or []),
    ]


# ===================================================================
# SCOUT MODE
# ===================================================================
def scout_news(cache, portfolio_keys, watchlist):
    items = fetch_news(portfolio_keys)
    seen  = set(cache.get("seen_news", []))
    new   = [i for i in items if i["id"] not in seen]
    cache["seen_news"] = list(seen | {i["id"] for i in items})
    if not new:
        return {"urgent": [], "candidates": []}
    items_str = "\n".join(f"- {i['title']}" for i in new[:60])
    prompt = (
        "Analyze headlines for stock impact.\n"
        f"Portfolio: {', '.join(portfolio_keys)}\n"
        f"Watchlist: {', '.join(watchlist[:60])}\n\n"
        f"News:\n{items_str}\n\n"
        'Return JSON: {"urgent":[{"ticker":"MU","headline":"...","why":"...","severity":"critical|high"}],'
        '"candidates":[{"ticker":"AMD","thesis":"<1 line>","trigger":"<news>"}]}\n\n'
        "Urgent = >5% move TODAY. Max 5 candidates. JSON only."
    )
    raw = call_gemini(prompt, max_tokens=900, json_mode=True)
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"news parse err: {e}")
        return {"urgent": [], "candidates": []}


def analyze_candidate(c, cache, portfolio_value):
    ticker = c["ticker"].upper()
    fund   = get_fundamentals(ticker, cache)
    val    = pick_valuation(fund) if fund else None
    tech   = technical_signals(ticker)
    sizing = None
    if val and val.get("current_price"):
        sizing = suggest_position_size(ticker, val["current_price"], portfolio_value)
    return {
        "ticker":    ticker,
        "thesis":    c.get("thesis",  ""),
        "trigger":   c.get("trigger", ""),
        "valuation": val,
        "tech":      tech,
        "sizing":    sizing,
    }


def fmt_candidate_html(a):
    val, tech, sizing = a.get("valuation"), a.get("tech"), a.get("sizing")
    verdict = tech_verdict(tech)

    if val and val["verdict"] == "undervalued" and verdict.startswith("oversold"):
        action, show_size = "✅ STRONG BUY SETUP", True
    elif val and val["verdict"] == "undervalued":
        action, show_size = "🟢 UNDERVALUED — wait for technicals", False
    elif verdict.startswith("oversold") and val and val["verdict"] == "fair":
        action, show_size = "🟡 TECHNICAL SETUP", False
    elif val and val["verdict"] == "overvalued" and "overbought" in verdict:
        action, show_size = "🔴 RICH + EXTENDED", False
    else:
        return None

    parts = [f"{code('$' + a['ticker'])} — {bold(action)}"]
    if a.get("thesis"):
        parts.append(f"<i>{esc(a['thesis'])}</i>")
    if val:
        parts.append(
            f"\n{bold('Fair value:')} {code('$' + str(val['fair_value']))} "
            f"({code('{:+.1f}%'.format(val['upside_pct']))}, {esc(val['method'])})"
        )
    parts.append(f"{bold('Technical:')} <i>{esc(verdict)}</i>")
    if tech:
        parts.append(f"Range: {code('$' + str(tech['support']))}–{code('$' + str(tech['resistance']))}")
    if show_size and sizing:
        parts.append(
            f"\n{bold('Suggested size:')} {code('$' + str(int(sizing['total_dollars'])))} "
            f"({code('{}%'.format(sizing['as_pct_portfolio']))} of portfolio, "
            f"{esc('{}% vol'.format(sizing['vol_annualized_pct']))})"
        )
        parts.append(
            f"Tranches: {code('$' + str(int(sizing['tranche_1'])))} / "
            f"{code('$' + str(int(sizing['tranche_2'])))} / "
            f"{code('$' + str(int(sizing['tranche_3'])))}"
        )
    if a.get("trigger"):
        parts.append(f"\n<i>Trigger: {esc(a['trigger'])}</i>")
    parts.append("<i>⚠️ Verify before acting.</i>")
    return "\n".join(parts)


# ===================================================================
# MAIN MODES
# ===================================================================
def run_scout():
    cache     = load_cache()
    portfolio = load_portfolio()
    portfolio = process_telegram_messages(cache, portfolio)

    portfolio_value = compute_portfolio_value(portfolio, cache)
    dd_alert = check_drawdown_alert(portfolio_value, cache)
    if dd_alert:
        send_drawdown_alert(dd_alert)

    keys      = list(portfolio.keys())
    watchlist = sorted(set(sum(WATCHLIST_BY_SECTOR.values(), [])) | set(keys))
    result    = scout_news(cache, keys, watchlist)
    sections  = []

    if result.get("urgent"):
        parts = [f"🚨 {bold('URGENT')}"]
        for u in result["urgent"]:
            sev = "🔴" if u.get("severity") == "critical" else "🟠"
            parts.append(
                f"\n{sev} {code('$' + u['ticker'])} — {esc(u['headline'])}"
                f"\n<i>{esc(u['why'])}</i>"
            )
        sections.append("\n".join(parts))

    if result.get("candidates"):
        analyzed = [analyze_candidate(c, cache, portfolio_value) for c in result["candidates"][:5]]
        recs = [r for r in (fmt_candidate_html(a) for a in analyzed) if r]
        if recs:
            sections.append(f"🔍 {bold('CANDIDATES')}\n\n" + "\n\n———\n\n".join(recs))

    save_cache(cache)
    save_portfolio(portfolio)
    if sections:
        send_chunked(sections)


def run_deep():
    cache     = load_cache()
    portfolio = load_portfolio()
    portfolio = process_telegram_messages(cache, portfolio)

    portfolio_value = compute_portfolio_value(portfolio, cache)
    dd_alert = check_drawdown_alert(portfolio_value, cache)
    if dd_alert:
        send_drawdown_alert(dd_alert)

    brief_data = deep_research_data(cache, portfolio)
    brief_data["watchlist_opportunities"] = enrich_watchlist_opps(
        brief_data.get("watchlist_opportunities") or [], cache
    )
    save_cache(cache)
    save_portfolio(portfolio)
    send_chunked(build_brief_sections(brief_data))


def run_listen():
    cache     = load_cache()
    portfolio = load_portfolio()
    portfolio = process_telegram_messages(cache, portfolio)

    portfolio_value = compute_portfolio_value(portfolio, cache)
    dd_alert = check_drawdown_alert(portfolio_value, cache)
    if dd_alert:
        send_drawdown_alert(dd_alert)

    save_cache(cache)
    save_portfolio(portfolio)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scout"
    try:
        if mode == "deep":
            run_deep()
        elif mode == "listen":
            run_listen()
        else:
            run_scout()
    except Exception as e:
        print(f"FATAL ({mode}): {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

