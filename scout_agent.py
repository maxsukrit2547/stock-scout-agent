import os
import json
import sys
import hashlib
import subprocess
from datetime import datetime
from google import genai
from google.genai import types
import feedparser
import requests
import yfinance as yf
import pandas as pd
import numpy as np

# Default portfolio - used only on first run if portfolio.json doesn't exist yet.
# After that, edits happen via Telegram messages.
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

WATCHLIST_BY_SECTOR = {
    "Semis":         ["AMD","INTC","MRVL","QCOM","ARM","ASML","AMAT","LRCX","KLAC","SMCI"],
    "Cloud_SaaS":    ["NOW","PANW","ZS","DDOG","SNOW","MDB","NET","OKTA","WDAY","ADBE"],
    "Internet":      ["NFLX","SPOT","RDDT","UBER","ABNB","BKNG"],
    "Space_Defense": ["LMT","RTX","BA","GD","PLTR","BKSY","PL","KTOS"],
    "Quantum_AI":    ["RGTI","QBTS","AI","BBAI","SOUN"],
    "EV_Auto":       ["RIVN","LCID","F","GM"],
    "Sector_ETFs":   ["XLK","SOXX","IGV","ARKX","XLC"],
}

RSS_FEEDS_BASE = [
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "https://www.cnbc.com/id/19854910/device/rss/rss.html",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://seekingalpha.com/market_currents.xml",
]

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# PORTFOLIO PERSISTENCE
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"portfolio load err: {e}, using defaults")
    return dict(DEFAULT_PORTFOLIO)


def save_portfolio(p):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(p, f, indent=2, sort_keys=True)


# CACHE
def load_cache():
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"seen_news": [], "fundamentals": {}, "last_telegram_offset": 0}


def save_cache(c):
    c["seen_news"] = c["seen_news"][-2000:]
    with open(CACHE_FILE, "w") as f:
        json.dump(c, f, default=str)


# NEWS
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
                items.append({
                    "id": hashlib.md5(e.link.encode()).hexdigest(),
                    "title": e.title,
                    "summary": e.get("summary", "")[:300],
                })
        except Exception as ex:
            print(f"feed err {url}: {ex}")
    return items


def fetch_ticker_news(ticker, limit=10):
    items = []
    try:
        feed = feedparser.parse(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US")
        for e in feed.entries[:limit]:
            items.append(f"- {e.title}")
    except Exception as ex:
        print(f"ticker news err {ticker}: {ex}")
    return "\n".join(items) if items else "(no recent news)"


# FUNDAMENTALS
def get_fundamentals(ticker, cache):
    key = ticker.upper()
    cached = cache["fundamentals"].get(key)
    now = datetime.utcnow().timestamp()
    if cached and (now - cached.get("fetched_at", 0)) < 86400:
        return cached
    try:
        info = yf.Ticker(ticker).info
        data = {
            "ticker": key,
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "pe": info.get("trailingPE"),
            "fwd_pe": info.get("forwardPE"),
            "ps": info.get("priceToSalesTrailing12Months"),
            "pb": info.get("priceToBook"),
            "ev_rev": info.get("enterpriseToRevenue"),
            "div_yield": info.get("dividendYield") or 0,
            "div_rate": info.get("dividendRate") or 0,
            "growth": info.get("earningsGrowth"),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap"),
            "fetched_at": now,
        }
        cache["fundamentals"][key] = data
        return data
    except Exception as e:
        print(f"yf fund err {ticker}: {e}")
        return None


def valuation_score(fund):
    if not fund or not fund.get("price"):
        return None
    pe = fund.get("pe")
    ps = fund.get("ps")
    pb = fund.get("pb")
    price = fund["price"]
    flags = []
    score = 0
    if pe and pe > 0:
        if pe < NORMAL_MULTIPLES["pe"] * 0.7:
            flags.append("PE_low")
            score -= 1
        if pe > NORMAL_MULTIPLES["pe"] * 1.5:
            flags.append("PE_high")
            score += 1
    if ps and ps > 0:
        if ps < NORMAL_MULTIPLES["ps"] * 0.7:
            flags.append("PS_low")
            score -= 1
        if ps > NORMAL_MULTIPLES["ps"] * 1.5:
            flags.append("PS_high")
            score += 1
    if pb and pb > 0:
        if pb < NORMAL_MULTIPLES["pb"] * 0.7:
            flags.append("PB_low")
            score -= 1
        if pb > NORMAL_MULTIPLES["pb"] * 1.5:
            flags.append("PB_high")
            score += 1

    verdict = "fair"
    if score <= -2:
        verdict = "undervalued"
    elif score >= 2:
        verdict = "overvalued"

    fair_estimates = []
    if pe and pe > 0:
        fair_estimates.append(price * NORMAL_MULTIPLES["pe"] / pe)
    if ps and ps > 0:
        fair_estimates.append(price * NORMAL_MULTIPLES["ps"] / ps)
    if pb and pb > 0:
        fair_estimates.append(price * NORMAL_MULTIPLES["pb"] / pb)
    fair_value = float(np.mean(fair_estimates)) if fair_estimates else None
    upside = ((fair_value - price) / price * 100) if fair_value else None

    ddm_value = None
    if fund.get("div_rate") and fund.get("div_rate") > 0:
        d1 = fund["div_rate"] * 1.05
        ddm_value = round(d1 / (0.09 - 0.05), 2)

    return {
        "verdict": verdict,
        "fair_value": round(fair_value, 2) if fair_value else None,
        "ddm_value": ddm_value,
        "upside_pct": round(upside, 1) if upside else None,
        "current_price": round(price, 2),
        "pe": round(pe, 2) if pe else None,
        "ps": round(ps, 2) if ps else None,
        "pb": round(pb, 2) if pb else None,
        "flags": flags,
    }


# TECHNICALS
def technical_signals(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="3mo", interval="1d")
        if hist.empty or len(hist) < 50:
            return None
        close = hist["Close"]

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_now = float(rsi.iloc[-1])

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        hist_macd = macd_line - signal_line

        ma50 = close.rolling(50).mean()
        ma20 = close.rolling(20).mean()
        above_ma50 = bool(close.iloc[-1] > ma50.iloc[-1])
        above_ma20 = bool(close.iloc[-1] > ma20.iloc[-1])

        macd_bullish = hist_macd.iloc[-1] > 0 and hist_macd.iloc[-1] > hist_macd.iloc[-2]
        macd_bearish = hist_macd.iloc[-1] < 0 and hist_macd.iloc[-1] < hist_macd.iloc[-2]

        support = float(close.tail(20).min())
        resistance = float(close.tail(20).max())

        if rsi_now < 40 and macd_bullish and above_ma50:
            signal = "BUY_SETUP"
        elif rsi_now < 35 and macd_bullish:
            signal = "BUY_SPECULATIVE"
        elif rsi_now > 70 and macd_bearish:
            signal = "EXIT_SETUP"
        else:
            signal = "WAIT"

        return {
            "rsi": round(rsi_now, 1),
            "macd_hist": round(float(hist_macd.iloc[-1]), 3),
            "above_ma50": above_ma50,
            "above_ma20": above_ma20,
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "signal": signal,
        }
    except Exception as e:
        print(f"ta err {ticker}: {e}")
        return None


# GEMINI
def call_gemini(prompt, max_tokens=600, json_mode=False, model=None):
    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=0.3,
        response_mime_type="application/json" if json_mode else None,
    )
    response = client.models.generate_content(
        model=model or MODEL_LIGHT,
        contents=prompt,
        config=config,
    )
    return response.text


# TELEGRAM
def telegram_send(text):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat = os.environ["TELEGRAM_CHAT_ID"]

    # Smart chunking - split at line boundaries to avoid breaking markdown
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > 3800:
            if current:
                chunks.append(current)
                current = line
            else:
                chunks.append(line[:3800])
                current = line[3800:]
        else:
            current = (current + "\n" + line) if current else line
    if current:
        chunks.append(current)

    for chunk in chunks:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat, "text": chunk, "parse_mode": "Markdown"},
                timeout=10,
            )
            # If markdown parse fails, retry as plain text
            if r.status_code != 200:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    data={"chat_id": chat, "text": chunk},
                    timeout=10,
                )
        except Exception as e:
            print(f"tg send err: {e}")


def telegram_get_updates(offset=None):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    params = {}
    if offset:
        params["offset"] = offset
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{token}/getUpdates",
            params=params,
            timeout=10,
        )
        return r.json().get("result", [])
    except Exception as e:
        print(f"tg get_updates err: {e}")
        return []


# TRADE PARSING
def parse_trade_command(message_text, current_tickers):
    prompt = (
        "Parse this user message about a stock trade into structured JSON.\n\n"
        f"User's current portfolio tickers: {', '.join(current_tickers)}\n"
        f'Message: "{message_text}"\n\n'
        "Possible actions:\n"
        '- "buy": user bought shares of a stock (new position or adding to existing)\n'
        '- "sell": user sold shares (can be percentage or absolute share count)\n'
        '- "set": user is directly overriding shares + avg_cost (for manual corrections)\n'
        '- "remove": user is removing/closing a position entirely\n'
        '- "ignore": message is not about a trade (greetings, questions, random text)\n\n'
        "Return JSON in this exact format:\n"
        "{\n"
        '  "action": "buy|sell|set|remove|ignore",\n'
        '  "ticker": "MU",\n'
        '  "shares": 1.5,\n'
        '  "is_percentage": false,\n'
        '  "price": 750.50,\n'
        '  "new_avg_cost": null,\n'
        '  "new_shares": null,\n'
        '  "confidence": 0.95,\n'
        '  "summary": "short human-readable summary of parsed action"\n'
        "}\n\n"
        "Rules:\n"
        '- For sell with percentage like "sold 30%", set shares=30 and is_percentage=true\n'
        '- For sell with shares like "sold 5 MU", set shares=5 and is_percentage=false\n'
        '- For "sold all", set is_percentage=true and shares=100\n'
        '- For buy, shares is the count bought, price is execution price\n'
        '- For set, fill new_shares and new_avg_cost\n'
        '- ticker should be uppercase US stock symbol\n'
        '- confidence < 0.7 means uncertain, low confidence trades will be ignored\n'
        '- If message is unclear or not a trade, action must be "ignore"\n\n'
        "JSON only, no other text."
    )

    raw = call_gemini(prompt, max_tokens=400, json_mode=True)
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"trade parse err: {e}")
        return {"action": "ignore", "confidence": 0}


def apply_trade(portfolio, trade):
    action = trade.get("action")
    confidence = trade.get("confidence", 0)
    if action == "ignore" or confidence < 0.7:
        return portfolio, None

    ticker = (trade.get("ticker") or "").upper()
    if not ticker:
        return portfolio, None

    p = {k: dict(v) for k, v in portfolio.items()}

    if action == "sell":
        if ticker not in p:
            return portfolio, f"⚠ Cannot sell {ticker} - not in portfolio"
        if trade.get("is_percentage"):
            pct = (trade.get("shares") or 0) / 100
            if pct >= 1:
                old = p.pop(ticker)
                return p, f"✓ Sold all of {ticker} (was {old['shares']:.4f} sh @ ${old['avg_cost']})"
            old_sh = p[ticker]["shares"]
            new_sh = round(old_sh * (1 - pct), 6)
            p[ticker]["shares"] = new_sh
            return p, f"✓ Sold {pct*100:.0f}% of {ticker}: {old_sh:.4f} → {new_sh:.4f} sh @ avg ${p[ticker]['avg_cost']}"
        else:
            sold = trade.get("shares") or 0
            old_sh = p[ticker]["shares"]
            new_sh = round(old_sh - sold, 6)
            if new_sh <= 0:
                p.pop(ticker)
                return p, f"✓ Sold {sold} sh of {ticker} (closed position)"
            p[ticker]["shares"] = new_sh
            return p, f"✓ Sold {sold} sh of {ticker}: {old_sh:.4f} → {new_sh:.4f} remaining"

    elif action == "buy":
        new_sh = trade.get("shares") or 0
        price = trade.get("price")
        if not new_sh or not price:
            return portfolio, "⚠ Buy needs both share count and price"
        if ticker in p:
            old_sh = p[ticker]["shares"]
            old_cost = p[ticker]["avg_cost"]
            total = old_sh + new_sh
            new_avg = (old_sh * old_cost + new_sh * price) / total
            p[ticker] = {"shares": round(total, 6), "avg_cost": round(new_avg, 2)}
            return p, f"✓ Bought {new_sh} {ticker} @ ${price}. New: {total:.4f} sh @ avg ${new_avg:.2f}"
        else:
            p[ticker] = {"shares": round(new_sh, 6), "avg_cost": round(price, 2)}
            return p, f"✓ New position: {new_sh} {ticker} @ ${price}"

    elif action == "set":
        ns = trade.get("new_shares")
        nc = trade.get("new_avg_cost")
        if ns is None or nc is None:
            return portfolio, "⚠ Set needs both shares and avg_cost"
        p[ticker] = {"shares": round(ns, 6), "avg_cost": round(nc, 2)}
        return p, f"✓ Set {ticker} to {ns} sh @ avg ${nc}"

    elif action == "remove":
        if ticker in p:
            p.pop(ticker)
            return p, f"✓ Removed {ticker}"
        return portfolio, f"⚠ {ticker} not in portfolio"

    return portfolio, None


def git_commit_portfolio():
    """Commit portfolio.json change back to repo."""
    try:
        subprocess.run(["git", "config", "user.name", "scout-agent"], check=True)
        subprocess.run(["git", "config", "user.email", "scout@bot.local"], check=True)
        subprocess.run(["git", "add", PORTFOLIO_FILE], check=True)
        result = subprocess.run(["git", "diff", "--staged", "--quiet"])
        if result.returncode == 0:
            return  # no changes
        subprocess.run(["git", "commit", "-m", "Update portfolio from Telegram"], check=True)
        subprocess.run(["git", "push"], check=True)
    except Exception as e:
        print(f"git commit err: {e}")


def process_telegram_messages(cache, portfolio):
    last_offset = cache.get("last_telegram_offset", 0)
    new_offset = last_offset + 1 if last_offset else None
    updates = telegram_get_updates(offset=new_offset)
    if not updates:
        return portfolio, False

    p = portfolio
    confirmations = []
    changed = False

    for u in updates:
        cache["last_telegram_offset"] = u["update_id"]
        msg = u.get("message", {})
        text = (msg.get("text") or "").strip()
        if not text or text.startswith("/"):
            continue

        trade = parse_trade_command(text, list(p.keys()))
        new_p, conf = apply_trade(p, trade)
        if conf:
            p = new_p
            confirmations.append(conf)
            if not conf.startswith("⚠"):
                changed = True

    if changed:
        save_portfolio(p)
        git_commit_portfolio()

    if confirmations:
        telegram_send("📒 *Portfolio updates*\n\n" + "\n".join(confirmations))

    return p, changed


# SCOUT MODE
def scout_news(cache, portfolio_keys, watchlist):
    items = fetch_news(portfolio_keys)
    seen = set(cache["seen_news"])
    new = [i for i in items if i["id"] not in seen]
    cache["seen_news"] = list(seen | {i["id"] for i in items})
    if not new:
        return {"urgent": [], "candidates": []}

    items_str = "\n".join(f"- {i['title']}" for i in new[:60])
    prompt = (
        "Analyze headlines for stock impact.\n"
        f"Portfolio: {', '.join(portfolio_keys)}\n"
        f"Watchlist: {', '.join(watchlist[:60])}\n\n"
        f"News (last 4h):\n{items_str}\n\n"
        "Return JSON.\n\n"
        "'urgent': items causing >5%% move TODAY in any portfolio ticker. Format:\n"
        '{"ticker": "MU", "headline": "...", "why": "...", "severity": "critical|high"}\n\n'
        "'candidates': potential opportunities - under/overvalued setups, sector momentum, "
        "catalyst-driven entries. From portfolio, watchlist, OR new tickers (any S&P 500). Format:\n"
        '{"ticker": "AMD", "thesis": "<1 line>", "trigger": "<news>"}\n\n'
        "Max 5 candidates. Empty arrays if nothing.\n\n"
        'Format: {"urgent": [...], "candidates": [...]}. JSON only.'
    )

    raw = call_gemini(prompt, max_tokens=900, json_mode=True)
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"news parse err: {e}\n{raw[:500]}")
        return {"urgent": [], "candidates": []}


def analyze_candidate(c, cache):
    ticker = c["ticker"].upper()
    fund = get_fundamentals(ticker, cache)
    val = valuation_score(fund) if fund else None
    tech = technical_signals(ticker)
    return {
        "ticker": ticker,
        "thesis": c.get("thesis", ""),
        "trigger": c.get("trigger", ""),
        "valuation": val,
        "technical": tech,
    }


def decide_action(a):
    val = a.get("valuation")
    tech = a.get("technical")
    if not val:
        return None
    v = val.get("verdict")
    t = (tech or {}).get("signal", "WAIT")
    if v == "undervalued" and t == "BUY_SETUP":
        return "STRONG BUY SETUP"
    if v == "undervalued" and t == "BUY_SPECULATIVE":
        return "BUY (speculative)"
    if v == "undervalued":
        return "UNDERVALUED - wait for tech confirmation"
    if t == "BUY_SETUP" and v == "fair":
        return "TECHNICAL SETUP"
    if v == "overvalued" and t == "EXIT_SETUP":
        return "RICH + EXTENDED"
    return None


def format_recommendation(a, action):
    val = a.get("valuation")
    tech = a.get("technical") or {}
    msg = f"*{a['ticker']}* - {action}\n"
    if a["thesis"]:
        msg += f"_{a['thesis']}_\n"
    msg += f"\n*Valuation*\nPrice ${val['current_price']} | P/E {val['pe']} | P/S {val['ps']} | P/BV {val['pb']}\n"
    if val.get("fair_value"):
        msg += f"Fair value: ${val['fair_value']} ({val['upside_pct']:+.1f}%)\n"
    if val.get("ddm_value"):
        msg += f"DDM: ${val['ddm_value']}\n"
    if tech:
        ma_dir = ">" if tech["above_ma50"] else "<"
        msg += f"\n*Technical*\nRSI {tech['rsi']} | MACD hist {tech['macd_hist']:+.3f} | {ma_dir} MA50\n"
        msg += f"Support ${tech['support']} | Resistance ${tech['resistance']} | {tech['signal']}\n"
    if a["trigger"]:
        msg += f"\n_Trigger: {a['trigger']}_\n"
    msg += "\n_Verify before acting._"
    return msg


# DEEP RESEARCH
def deep_research(cache, portfolio):
    sections = []
    for ticker, d in portfolio.items():
        fund = get_fundamentals(ticker, cache)
        if not fund or not fund.get("price"):
            continue
        val = valuation_score(fund)
        tech = technical_signals(ticker)
        pl_pct = (fund["price"] - d["avg_cost"]) / d["avg_cost"] * 100
        news = fetch_ticker_news(ticker, limit=8)

        block = f"## {ticker} ({fund.get('industry') or fund.get('sector')})\n"
        block += f"Position: {d['shares']:.4f} sh @ avg ${d['avg_cost']} | Current ${fund['price']:.2f} | P&L {pl_pct:+.1f}%\n"
        block += f"Fundamentals: P/E {val['pe']} (fwd {fund.get('fwd_pe')}) | P/S {val['ps']} | P/BV {val['pb']} | Verdict: {val['verdict']}\n"
        block += f"Fair value: ${val['fair_value']} ({val['upside_pct']:+.1f}% from current)"
        if val.get("ddm_value"):
            block += f" | DDM: ${val['ddm_value']}"
        if tech:
            ma_pos = "Above" if tech["above_ma50"] else "Below"
            block += f"\nTechnicals: RSI {tech['rsi']} | MACD hist {tech['macd_hist']:+.3f} | {ma_pos} MA50 | Signal: {tech['signal']}"
            block += f"\nRecent range: ${tech['support']} to ${tech['resistance']}"
        block += f"\nRecent news:\n{news}"
        sections.append(block)

    full_context = "\n\n".join(sections)

    prompt = (
        "You are a portfolio analyst writing a pre-market deep research brief for a Bangkok-based "
        "investor, 2 hours before US market open. Their full portfolio data is below: current "
        "prices, cost basis, fundamentals, technicals, and last week's news per ticker.\n\n"
        f"{full_context}\n\n"
        "Write a comprehensive Markdown brief. STRICT REQUIREMENT: Total length must be under 1500 "
        "words and must include all 5 sections completely - do NOT cut off mid-section.\n\n"
        "## Executive Summary\n"
        "3-4 sentences: overall portfolio health, biggest risk for today, biggest opportunity.\n\n"
        "## Action Priorities (max 5, ranked)\n"
        "Specific, actionable items with concrete price levels. Cite actual numbers from data above.\n"
        "Example format: '1. Trim MU 20% above $760 - 2nd tranche, P/E stretched at 32.'\n\n"
        "## Per-Stock Status (all positions)\n"
        "Format: TICKER green/yellow/red - 1 line state + key catalyst this week + action.\n"
        "Keep each stock to ONE LINE only.\n\n"
        "## Top 3 Risks Today\n"
        "Portfolio-wide risks: macro, sector, or specific names. 1 line each.\n\n"
        "## Watch at Open\n"
        "2-3 specific tickers and what to look for in the first 30 min of trading.\n\n"
        "Tone: direct, specific, no filler, no disclaimers. Finish all 5 sections."
    )

    return call_gemini(prompt, max_tokens=8000, model=MODEL_DEEP)


# MAIN
def run_scout():
    cache = load_cache()
    portfolio = load_portfolio()

    # Process any new Telegram trade commands first
    portfolio, _ = process_telegram_messages(cache, portfolio)

    portfolio_keys = list(portfolio.keys())
    watchlist = sorted(set(sum(WATCHLIST_BY_SECTOR.values(), [])) | set(portfolio_keys))

    result = scout_news(cache, portfolio_keys, watchlist)
    msgs = []
    if result.get("urgent"):
        msg = "*URGENT*\n\n"
        for u in result["urgent"]:
            sev = "[CRIT]" if u.get("severity") == "critical" else "[HIGH]"
            msg += f"{sev} *{u['ticker']}* - {u['headline']}\n_{u['why']}_\n\n"
        msgs.append(msg)
    if result.get("candidates"):
        analyzed = [analyze_candidate(c, cache) for c in result["candidates"][:5]]
        recs = []
        for a in analyzed:
            action = decide_action(a)
            if action:
                recs.append(format_recommendation(a, action))
        if recs:
            msgs.append("*Candidates*\n\n" + "\n\n---\n\n".join(recs))

    save_cache(cache)
    save_portfolio(portfolio)  # ensure portfolio.json exists
    for m in msgs:
        telegram_send(m)


def run_deep():
    cache = load_cache()
    portfolio = load_portfolio()

    # Process any new Telegram trade commands first
    portfolio, _ = process_telegram_messages(cache, portfolio)

    brief = deep_research(cache, portfolio)
    save_cache(cache)
    save_portfolio(portfolio)
    header = f"*Deep Research - {datetime.utcnow().strftime('%b %d %Y')}*\n_2 hours to US open_\n\n"
    telegram_send(header + brief)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scout"
    if mode == "deep":
        run_deep()
    else:
        run_scout()
