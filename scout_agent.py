"""
Stock Scout Agent
- Scout mode: news scan every 4h on weekdays
- Deep mode: structured pre-market brief at 18:30 BKK on weekdays
- Listens to Telegram messages for portfolio updates (sold/bought/set/remove)
- Renders to Telegram using HTML parse mode with safe section-aware chunking
"""

import os
import sys
import json
import html
import hashlib
import subprocess
from datetime import datetime, timedelta

import requests
import feedparser
import yfinance as yf
import pandas as pd
import numpy as np
from google import genai
from google.genai import types


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
MODEL_DEEP = "gemini-2.5-flash"

NORMAL_MULTIPLES = {"pe": 22, "ps": 6, "pb": 4}
TG_SAFE_LIMIT = 3800  # under Telegram's 4096 hard cap

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


# ===================================================================
# PERSISTENCE: portfolio + cache
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
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(p, f, indent=2, sort_keys=True)


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


# ===================================================================
# DATA: news, fundamentals, technicals
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
        feed = feedparser.parse(
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        )
        for e in feed.entries[:limit]:
            items.append(f"- {e.title}")
    except Exception as ex:
        print(f"ticker news err {ticker}: {ex}")
    return "\n".join(items) if items else "(no recent news)"


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
            "div_yield": info.get("dividendYield") or 0,
            "div_rate": info.get("dividendRate") or 0,
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


# ===================================================================
# ANALYSIS: valuation, technical signals
# ===================================================================

def valuation_score(fund):
    if not fund or not fund.get("price"):
        return None
    pe, ps, pb, price = fund.get("pe"), fund.get("ps"), fund.get("pb"), fund["price"]
    flags, score = [], 0
    if pe and pe > 0:
        if pe < NORMAL_MULTIPLES["pe"] * 0.7:
            flags.append("PE_low"); score -= 1
        if pe > NORMAL_MULTIPLES["pe"] * 1.5:
            flags.append("PE_high"); score += 1
    if ps and ps > 0:
        if ps < NORMAL_MULTIPLES["ps"] * 0.7:
            flags.append("PS_low"); score -= 1
        if ps > NORMAL_MULTIPLES["ps"] * 1.5:
            flags.append("PS_high"); score += 1
    if pb and pb > 0:
        if pb < NORMAL_MULTIPLES["pb"] * 0.7:
            flags.append("PB_low"); score -= 1
        if pb > NORMAL_MULTIPLES["pb"] * 1.5:
            flags.append("PB_high"); score += 1

    verdict = "fair"
    if score <= -2: verdict = "undervalued"
    elif score >= 2: verdict = "overvalued"

    fair_estimates = []
    if pe and pe > 0: fair_estimates.append(price * NORMAL_MULTIPLES["pe"] / pe)
    if ps and ps > 0: fair_estimates.append(price * NORMAL_MULTIPLES["ps"] / ps)
    if pb and pb > 0: fair_estimates.append(price * NORMAL_MULTIPLES["pb"] / pb)
    fair_value = float(np.mean(fair_estimates)) if fair_estimates else None
    upside = ((fair_value - price) / price * 100) if fair_value else None

    ddm_value = None
    if fund.get("div_rate") and fund.get("div_rate") > 0:
        d1 = fund["div_rate"] * 1.05
        ddm_value = round(d1 / 0.04, 2)

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
        above_ma50 = bool(close.iloc[-1] > ma50.iloc[-1])

        macd_bull = hist_macd.iloc[-1] > 0 and hist_macd.iloc[-1] > hist_macd.iloc[-2]
        macd_bear = hist_macd.iloc[-1] < 0 and hist_macd.iloc[-1] < hist_macd.iloc[-2]

        support = float(close.tail(20).min())
        resistance = float(close.tail(20).max())

        if rsi_now < 40 and macd_bull and above_ma50:
            sig = "BUY_SETUP"
        elif rsi_now < 35 and macd_bull:
            sig = "BUY_SPECULATIVE"
        elif rsi_now > 70 and macd_bear:
            sig = "EXIT_SETUP"
        else:
            sig = "WAIT"

        return {
            "rsi": round(rsi_now, 1),
            "macd_hist": round(float(hist_macd.iloc[-1]), 3),
            "above_ma50": above_ma50,
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "signal": sig,
        }
    except Exception as e:
        print(f"ta err {ticker}: {e}")
        return None


# ===================================================================
# GEMINI WRAPPER
# ===================================================================

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


# ===================================================================
# TELEGRAM: send + chunk (HTML parse mode)
# ===================================================================

def esc(s):
    """Escape user content for Telegram HTML parse mode."""
    if s is None:
        return ""
    return html.escape(str(s), quote=False)


def code(s):
    """Wrap in <code> for monospace."""
    return f"<code>{esc(s)}</code>"


def bold(s):
    return f"<b>{esc(s)}</b>"


def chunk_sections(sections, limit=TG_SAFE_LIMIT):
    """
    Pack section strings into messages without splitting any section.
    sections: list of strings, each one a complete logical block.
    Returns: list of message strings, each <= limit.
    """
    messages = []
    current = ""
    for sec in sections:
        if not sec:
            continue
        # If section alone exceeds limit, hard-split at line boundaries
        if len(sec) > limit:
            if current:
                messages.append(current)
                current = ""
            for sub in _split_at_lines(sec, limit):
                messages.append(sub)
            continue
        # Try to append to current message
        sep = "\n\n" if current else ""
        if len(current) + len(sep) + len(sec) <= limit:
            current = current + sep + sec
        else:
            messages.append(current)
            current = sec
    if current:
        messages.append(current)
    return messages


def _split_at_lines(text, limit):
    """Hard-split a single oversized section at line boundaries."""
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit:
            if cur:
                chunks.append(cur)
                cur = line
            else:
                # single line too long — slice
                chunks.append(line[:limit])
                cur = line[limit:]
        else:
            cur = cur + "\n" + line if cur else line
    if cur:
        chunks.append(cur)
    return chunks


def telegram_send(text, html_mode=True):
    """Send a single message. Falls back to plain text if HTML parsing fails."""
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat = os.environ["TELEGRAM_CHAT_ID"]
    payload = {
        "chat_id": chat,
        "text": text,
        "disable_web_page_preview": True,
    }
    if html_mode:
        payload["parse_mode"] = "HTML"
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload,
            timeout=15,
        )
        if r.status_code != 200 and html_mode:
            print(f"tg HTML failed {r.status_code}: {r.text[:200]} — retrying as plain text")
            payload.pop("parse_mode", None)
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data=payload,
                timeout=15,
            )
    except Exception as e:
        print(f"tg send err: {e}")


def send_chunked(sections):
    """Build chunked messages from sections list and send each."""
    for msg in chunk_sections(sections):
        telegram_send(msg, html_mode=True)


# ===================================================================
# TELEGRAM: receive (trade commands)
# ===================================================================

def telegram_get_updates(offset=None):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    params = {"offset": offset} if offset else {}
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


def parse_trade_command(message_text, current_tickers):
    prompt = (
        "Parse this user message about a stock trade into structured JSON.\n\n"
        f"Current portfolio: {', '.join(current_tickers)}\n"
        f'Message: "{message_text}"\n\n'
        "Actions:\n"
        '- "buy": bought shares of a stock\n'
        '- "sell": sold shares (percentage or absolute)\n'
        '- "set": override shares + avg_cost directly\n'
        '- "remove": close position entirely\n'
        '- "ignore": not a trade message\n\n'
        "Return JSON:\n"
        "{\n"
        '  "action": "buy|sell|set|remove|ignore",\n'
        '  "ticker": "MU",\n'
        '  "shares": 1.5,\n'
        '  "is_percentage": false,\n'
        '  "price": 750.50,\n'
        '  "new_avg_cost": null,\n'
        '  "new_shares": null,\n'
        '  "confidence": 0.95,\n'
        '  "summary": "short summary"\n'
        "}\n\n"
        'For "sold 30%": shares=30, is_percentage=true.\n'
        'For "sold all X": is_percentage=true, shares=100.\n'
        'For "bought N at P": shares=N, price=P, is_percentage=false.\n'
        "Confidence < 0.7 will be ignored. JSON only."
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
        sold = trade.get("shares") or 0
        old_sh = p[ticker]["shares"]
        new_sh = round(old_sh - sold, 6)
        if new_sh <= 0:
            p.pop(ticker)
            return p, f"✅ Sold {sold} sh of {ticker} (closed)"
        p[ticker]["shares"] = new_sh
        return p, f"✅ Sold {sold} {ticker}: {old_sh:.4f} → {new_sh:.4f} remaining"

    if a == "buy":
        new_sh = trade.get("shares") or 0
        price = trade.get("price")
        if not new_sh or not price:
            return portfolio, "⚠️ Buy needs share count and price"
        if ticker in p:
            old_sh = p[ticker]["shares"]
            old_cost = p[ticker]["avg_cost"]
            total = old_sh + new_sh
            new_avg = (old_sh * old_cost + new_sh * price) / total
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
        subprocess.run(["git", "config", "user.name", "scout-agent"], check=True)
        subprocess.run(["git", "config", "user.email", "scout@bot.local"], check=True)
        subprocess.run(["git", "add", PORTFOLIO_FILE], check=True)
        if subprocess.run(["git", "diff", "--staged", "--quiet"]).returncode == 0:
            return
        subprocess.run(["git", "commit", "-m", "Update portfolio from Telegram"], check=True)
        subprocess.run(["git", "push"], check=True)
    except Exception as e:
        print(f"git commit err: {e}")


def process_telegram_messages(cache, portfolio):
    last_offset = cache.get("last_telegram_offset", 0)
    updates = telegram_get_updates(offset=last_offset + 1 if last_offset else None)
    if not updates:
        return portfolio
    p = portfolio
    confirmations, changed = [], False
    for u in updates:
        cache["last_telegram_offset"] = u["update_id"]
        text = ((u.get("message") or {}).get("text") or "").strip()
        if not text or text.startswith("/"):
            continue
        trade = parse_trade_command(text, list(p.keys()))
        new_p, conf = apply_trade(p, trade)
        if conf:
            p = new_p
            confirmations.append(conf)
            if not conf.startswith("⚠️"):
                changed = True
    if changed:
        save_portfolio(p)
        git_commit_portfolio()
    if confirmations:
        body = "\n".join(esc(c) for c in confirmations)
        telegram_send(f"📒 <b>Portfolio updates</b>\n\n{body}")
    return p


# ===================================================================
# DEEP RESEARCH: Gemini structured JSON brief
# ===================================================================

def deep_research_data(cache, portfolio):
    """Build structured JSON brief from Gemini."""
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
        block += f"Position: {d['shares']:.4f} sh @ ${d['avg_cost']} | Now ${fund['price']:.2f} | P&L {pl_pct:+.1f}%\n"
        block += f"P/E {val['pe']} (fwd {fund.get('fwd_pe')}) | P/S {val['ps']} | P/BV {val['pb']} | {val['verdict']}\n"
        block += f"Fair value: ${val['fair_value']} ({val['upside_pct']:+.1f}%)"
        if val.get("ddm_value"):
            block += f" | DDM: ${val['ddm_value']}"
        if tech:
            ma_pos = "Above" if tech["above_ma50"] else "Below"
            block += f"\nRSI {tech['rsi']} | MACD {tech['macd_hist']:+.3f} | {ma_pos} MA50 | {tech['signal']}"
            block += f"\nRange: ${tech['support']}-${tech['resistance']}"
        block += f"\nNews:\n{news}"
        sections.append(block)

    full_context = "\n\n".join(sections)

    prompt = (
        "You are a portfolio analyst. Analyze the portfolio data below and return STRICT JSON.\n\n"
        f"{full_context}\n\n"
        "Return ONLY valid JSON in this exact structure:\n"
        "{\n"
        '  "executive_summary": {\n'
        '    "overall_status": "1-2 sentence portfolio health snapshot",\n'
        '    "risk": "biggest risk for today, 1 sentence",\n'
        '    "opportunity": "biggest opportunity today, 1 sentence"\n'
        "  },\n"
        '  "action_priorities": {\n'
        '    "trim_take_profits": [\n'
        '      {"ticker": "MU", "action": "Trim 20% above $760", "trigger": "P/E stretched, +253% return"}\n'
        "    ],\n"
        '    "monitor_for_exit": [\n'
        '      {"ticker": "ORCL", "concern": "thesis weakening", "watch_for": "Cloud guidance Q1"}\n'
        "    ]\n"
        "  },\n"
        '  "watch_at_open": [\n'
        '    {"ticker": "MU", "zone": "$745-755", "rsi": 78, "scenario": "Trim if gaps above $760"}\n'
        "  ],\n"
        '  "macro_risks": [\n'
        '    "Risk 1 in 1 line",\n'
        '    "Risk 2 in 1 line",\n'
        '    "Risk 3 in 1 line"\n'
        "  ],\n"
        '  "per_stock_status": {\n'
        '    "green": [{"ticker": "RKLB", "state": "+154% momentum intact", "catalyst": "next launch"}],\n'
        '    "yellow": [{"ticker": "MU", "state": "RSI 78 overbought", "catalyst": "trim watch"}],\n'
        '    "red": [{"ticker": "ORCL", "state": "-25% from cost", "catalyst": "Q1 May 22"}]\n'
        "  }\n"
        "}\n\n"
        "Categorization:\n"
        "- green: hold/add candidates with healthy momentum\n"
        "- yellow: watch closely, profit-taking candidates, mixed signals\n"
        "- red: thesis at risk, potential exit candidates\n\n"
        "Constraints:\n"
        "- watch_at_open: exactly 3 tickers most relevant for today's session\n"
        "- macro_risks: exactly 3 items\n"
        "- action_priorities total combined: max 5 items\n"
        "- per_stock_status: every portfolio ticker categorized exactly once\n"
        "- Keep all text fields concise; this will display on mobile\n\n"
        "JSON only, no commentary."
    )

    raw = call_gemini(prompt, max_tokens=8000, model=MODEL_DEEP, json_mode=True)
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"deep parse err: {e}")
        return {
            "executive_summary": {"overall_status": "Parse error", "risk": "—", "opportunity": "—"},
            "action_priorities": {"trim_take_profits": [], "monitor_for_exit": []},
            "watch_at_open": [],
            "macro_risks": [],
            "per_stock_status": {"green": [], "yellow": [], "red": []},
        }


# ===================================================================
# MESSAGE BUILDER: structured data → HTML sections
# ===================================================================

def build_header():
    bkk = datetime.utcnow() + timedelta(hours=7)
    return (
        f"📊 {bold('PRE-MARKET DEEP RESEARCH')}\n"
        f"<i>{esc(bkk.strftime('%b %d, %Y'))} · {esc(bkk.strftime('%H:%M'))} BKK · 2hr to US open</i>"
    )


def build_exec_summary(d):
    return (
        f"📑 {bold('EXECUTIVE SUMMARY')}\n\n"
        f"{bold('Status:')} {esc(d.get('overall_status', '—'))}\n\n"
        f"{bold('Top Risk:')} {esc(d.get('risk', '—'))}\n\n"
        f"{bold('Opportunity:')} {esc(d.get('opportunity', '—'))}"
    )


def build_action_priorities(d):
    parts = [f"🎯 {bold('ACTION PRIORITIES')}"]
    trim = d.get("trim_take_profits") or []
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
        if it.get("rsi"):
            line += f"   RSI: {code(it['rsi'])}"
        if it.get("scenario"):
            line += f"\n   <i>{esc(it['scenario'])}</i>"
        parts.append(line)
    return "\n".join(parts)


def build_macro_risks(risks):
    parts = [f"⚠️ {bold('TOP 3 MACRO RISKS')}"]
    if not risks:
        parts.append("\n<i>No material macro risks flagged.</i>")
    for i, r in enumerate(risks[:3], 1):
        parts.append(f"\n{i}. {esc(r)}")
    return "\n".join(parts)


def build_per_stock(d):
    parts = [f"📋 {bold('PER-STOCK CATALYSTS &amp; STATUS')}"]

    def add_group(label, emoji, items):
        if not items:
            return
        parts.append(f"\n{emoji} {bold(label)}")
        for it in items:
            line = f"• {code('$' + it['ticker'])}: {esc(it.get('state', ''))}"
            if it.get("catalyst"):
                line += f" · <i>{esc(it['catalyst'])}</i>"
            parts.append(line)

    add_group("Green", "🟢", d.get("green") or [])
    add_group("Yellow", "🟡", d.get("yellow") or [])
    add_group("Red", "🔴", d.get("red") or [])
    return "\n".join(parts)


def build_brief_sections(brief_data):
    """Compose ordered list of HTML section strings ready for chunking."""
    return [
        build_header(),
        build_exec_summary(brief_data.get("executive_summary") or {}),
        build_action_priorities(brief_data.get("action_priorities") or {}),
        build_watch_at_open(brief_data.get("watch_at_open") or []),
        build_macro_risks(brief_data.get("macro_risks") or []),
        build_per_stock(brief_data.get("per_stock_status") or {}),
    ]


# ===================================================================
# SCOUT MODE: news + candidate alerts
# ===================================================================

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
        "Return JSON:\n"
        '{"urgent":[{"ticker":"MU","headline":"...","why":"...","severity":"critical|high"}],'
        '"candidates":[{"ticker":"AMD","thesis":"<1 line>","trigger":"<news>"}]}\n\n'
        "Urgent = >5%% move TODAY. Max 5 candidates. Empty arrays if nothing. JSON only."
    )
    raw = call_gemini(prompt, max_tokens=900, json_mode=True)
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"news parse err: {e}")
        return {"urgent": [], "candidates": []}


def analyze_candidate(c, cache):
    ticker = c["ticker"].upper()
    fund = get_fundamentals(ticker, cache)
    return {
        "ticker": ticker,
        "thesis": c.get("thesis", ""),
        "trigger": c.get("trigger", ""),
        "valuation": valuation_score(fund) if fund else None,
        "technical": technical_signals(ticker),
    }


def decide_action(a):
    val, tech = a.get("valuation"), a.get("technical")
    if not val:
        return None
    v, t = val.get("verdict"), (tech or {}).get("signal", "WAIT")
    if v == "undervalued" and t == "BUY_SETUP": return "✅ STRONG BUY SETUP"
    if v == "undervalued" and t == "BUY_SPECULATIVE": return "🟢 BUY (speculative)"
    if v == "undervalued": return "🟢 UNDERVALUED — wait for tech"
    if t == "BUY_SETUP" and v == "fair": return "🟡 TECHNICAL SETUP"
    if v == "overvalued" and t == "EXIT_SETUP": return "🔴 RICH + EXTENDED"
    return None


def fmt_candidate_html(a, action):
    val, tech = a.get("valuation"), a.get("technical") or {}
    parts = [f"{code('$' + a['ticker'])} — {bold(action)}"]
    if a["thesis"]:
        parts.append(f"<i>{esc(a['thesis'])}</i>")
    parts.append(
        f"\n{bold('Valuation')}\n"
        f"Price {code('$' + str(val['current_price']))}  P/E {code(val['pe'])}  P/S {code(val['ps'])}  P/BV {code(val['pb'])}"
    )
    if val.get("fair_value"):
        parts.append(f"Fair value {code('$' + str(val['fair_value']))} ({code(f'{val.get('upside_pct'):+.1f}%')})")
    if tech:
        ma = ">" if tech["above_ma50"] else "<"
        parts.append(
            f"\n{bold('Technical')}\n"
            f"RSI {code(tech['rsi'])}  MACD {code(f'{tech['macd_hist']:+.3f}')}  {ma} MA50\n"
            f"Support {code('$' + str(tech['support']))}  Resistance {code('$' + str(tech['resistance']))}  {code(tech['signal'])}"
        )
    if a["trigger"]:
        parts.append(f"\n<i>Trigger: {esc(a['trigger'])}</i>")
    parts.append("<i>⚠️ Verify before acting.</i>")
    return "\n".join(parts)


# ===================================================================
# MAIN MODES
# ===================================================================

def run_scout():
    cache = load_cache()
    portfolio = load_portfolio()
    portfolio = process_telegram_messages(cache, portfolio)

    keys = list(portfolio.keys())
    watchlist = sorted(set(sum(WATCHLIST_BY_SECTOR.values(), [])) | set(keys))

    result = scout_news(cache, keys, watchlist)
    sections = []

    if result.get("urgent"):
        parts = [f"🚨 {bold('URGENT')}"]
        for u in result["urgent"]:
            sev = "🔴" if u.get("severity") == "critical" else "🟠"
            parts.append(
                f"\n{sev} {code('$' + u['ticker'])} — {esc(u['headline'])}\n"
                f"<i>{esc(u['why'])}</i>"
            )
        sections.append("\n".join(parts))

    if result.get("candidates"):
        analyzed = [analyze_candidate(c, cache) for c in result["candidates"][:5]]
        recs = [fmt_candidate_html(a, act) for a in analyzed if (act := decide_action(a))]
        if recs:
            sections.append(f"🔍 {bold('CANDIDATES')}\n\n" + "\n\n———\n\n".join(recs))

    save_cache(cache)
    save_portfolio(portfolio)

    if sections:
        send_chunked(sections)


def run_deep():
    cache = load_cache()
    portfolio = load_portfolio()
    portfolio = process_telegram_messages(cache, portfolio)

    brief_data = deep_research_data(cache, portfolio)
    save_cache(cache)
    save_portfolio(portfolio)

    sections = build_brief_sections(brief_data)
    send_chunked(sections)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scout"
    if mode == "deep":
        run_deep()
    else:
        run_scout()
