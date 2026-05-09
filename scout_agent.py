import os, json, sys, hashlib
from datetime import datetime
from google import genai
from google.genai import types
import feedparser
import requests
import yfinance as yf
import pandas as pd
import numpy as np

#===== EDIT THIS — your real positions =====
PORTFOLIO_DETAIL = {
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
PORTFOLIO = list(PORTFOLIO_DETAIL.keys())

Watchlist for opportunity scanning across sectors
WATCHLIST_BY_SECTOR = {
"Semis":         ["AMD","INTC","MRVL","QCOM","ARM","ASML","AMAT","LRCX","KLAC","SMCI"],
"Cloud_SaaS":    ["NOW","PANW","ZS","DDOG","SNOW","MDB","NET","OKTA","WDAY","ADBE"],
"Internet":      ["NFLX","SPOT","RDDT","UBER","ABNB","BKNG"],
"Space_Defense": ["LMT","RTX","BA","GD","PLTR","BKSY","PL","KTOS"],
"Quantum_AI":    ["RGTI","QBTS","AI","BBAI","SOUN"],
"EV_Auto":       ["RIVN","LCID","F","GM"],
"Sector_ETFs":   ["XLK","SOXX","IGV","ARKX","XLC"],
}
WATCHLIST = sorted(set(sum(WATCHLIST_BY_SECTOR.values(), [])) | set(PORTFOLIO))

RSS_FEEDS = [ "https://www.cnbc.com/id/100003114/device/rss/rss.html", "https://www.cnbc.com/id/10000664/device/rss/rss.html", "https://www.cnbc.com/id/19854910/device/rss/rss.html", "https://feeds.marketwatch.com/marketwatch/topstories/", "https://seekingalpha.com/market_currents.xml", ] for t in PORTFOLIO: RSS_FEEDS.append(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US")

CACHE_FILE = "scout_cache.json"
MODEL_LIGHT = "gemini-2.5-flash-lite"
MODEL_DEEP  = "gemini-2.5-flash"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

NORMAL_MULTIPLES = {"pe": 22, "ps": 6, "pb": 4}

===== CACHE =====
def load_cache():
try:
with open(CACHE_FILE) as f:
return json.load(f)
except FileNotFoundError:
return {"seen_news": [], "fundamentals": {}}

def save_cache(c):
c["seen_news"] = c["seen_news"][-2000:]
with open(CACHE_FILE, "w") as f:
json.dump(c, f, default=str)

===== NEWS =====
def fetch_news():
items = []
for url in RSS_FEEDS:
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

def fetch_ticker_news(ticker, limit=10): items = [] try: feed = feedparser.parse(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US") for e in feed.entries[:limit]: items.append(f"- {e.title}") except Exception as ex: print(f"ticker news err {ticker}: {ex}") return "\n".join(items) if items else "(no recent news)"

===== FUNDAMENTALS =====
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
pe, ps, pb = fund.get("pe"), fund.get("ps"), fund.get("pb")
price = fund["price"]
flags, score = [], 0
if pe and pe > 0:
if pe < NORMAL_MULTIPLES["pe"] * 0.7: flags.append("PE_low");  score -= 1
if pe > NORMAL_MULTIPLES["pe"] * 1.5: flags.append("PE_high"); score += 1
if ps and ps > 0:
if ps < NORMAL_MULTIPLES["ps"] * 0.7: flags.append("PS_low");  score -= 1
if ps > NORMAL_MULTIPLES["ps"] * 1.5: flags.append("PS_high"); score += 1
if pb and pb > 0:
if pb < NORMAL_MULTIPLES["pb"] * 0.7: flags.append("PB_low");  score -= 1
if pb > NORMAL_MULTIPLES["pb"] * 1.5: flags.append("PB_high"); score += 1

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
===== TECHNICALS =====
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
===== GEMINI =====
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

===== SCOUT MODE =====
def scout_news(cache):
items = fetch_news()
seen = set(cache["seen_news"])
new = [i for i in items if i["id"] not in seen]
cache["seen_news"] = list(seen | {i["id"] for i in items})
if not new:
return {"urgent": [], "candidates": []}

items_str = "\n".join(f"- {i['title']}" for i in new[:60])
prompt = f"""Analyze headlines for stock impact.
Portfolio: {', '.join(PORTFOLIO)}
Watchlist: {', '.join(WATCHLIST[:60])}

News (last 4h):
{items_str}

Return JSON.

"urgent": items causing >5% move TODAY in any portfolio ticker. Format:
{{"ticker": "MU", "headline": "...", "why": "...", "severity": "critical|high"}}

"candidates": potential opportunities — under/overvalued setups, sector momentum, catalyst-driven entries. From portfolio, watchlist, OR new tickers (any S&P 500). Format:
{{"ticker": "AMD", "thesis": "<1 line>", "trigger": "<news>"}}

Max 5 candidates. Empty arrays if nothing.

Format: {{"urgent": [...], "candidates": [...]}}. JSON only."""

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
val, tech = a.get("valuation"), a.get("technical")
if not val: return None
v, t = val.get("verdict"), (tech or {}).get("signal", "WAIT")
if v == "undervalued" and t == "BUY_SETUP":     return "✅ STRONG BUY SETUP"
if v == "undervalued" and t == "BUY_SPECULATIVE": return "🟢 BUY (speculative)"
if v == "undervalued":                            return "🟢 UNDERVALUED — wait for tech confirmation"
if t == "BUY_SETUP" and v == "fair":              return "🟡 TECHNICAL SETUP"
if v == "overvalued" and t == "EXIT_SETUP":       return "🔴 RICH + EXTENDED"
return None

def format_recommendation(a, action): val, tech = a.get("valuation"), a.get("technical") or {} msg = f"{a['ticker']} — {action}\n" if a["thesis"]: msg += f"{a['thesis']}\n" msg += f"\nValuation\nPrice ${val['current_price']} | P/E {val['pe']} | P/S {val['ps']} | P/BV {val['pb']}\n" if val.get("fair_value"): msg += f"Fair value: ${val['fair_value']} ({val['upside_pct']:+.1f}%)\n" if val.get("ddm_value"): msg += f"DDM: ${val['ddm_value']}\n" if tech: msg += f"\nTechnical\nRSI {tech['rsi']} | MACD hist {tech['macd_hist']:+.3f} | {'>' if tech['above_ma50'] else '<'} MA50\n" msg += f"Support ${tech['support']} | Resistance ${tech['resistance']} | {tech['signal']}\n" if a["trigger"]: msg += f"\n_Trigger: {a['trigger']}\n" msg += "\n⚠ Verify before acting._" return msg

===== DEEP RESEARCH =====
def deep_research(cache):
sections = []
for ticker in PORTFOLIO:
fund = get_fundamentals(ticker, cache)
if not fund or not fund.get("price"):
continue
val = valuation_score(fund)
tech = technical_signals(ticker)
d = PORTFOLIO_DETAIL[ticker]
pl_pct = (fund["price"] - d["avg_cost"]) / d["avg_cost"] * 100
news = fetch_ticker_news(ticker, limit=8)

    block = f"""## {ticker} ({fund.get('industry') or fund.get('sector')})
Position: {d['shares']:.4f} sh @ avg ${d['avg_cost']} | Current ${fund['price']:.2f} | P&L {pl_pct:+.1f}% Fundamentals: P/E {val['pe']} (fwd {fund.get('fwd_pe')}) | P/S {val['ps']} | P/BV {val['pb']} | Verdict: {val['verdict']} Fair value: ${val['fair_value']} ({val['upside_pct']:+.1f}% from current)""" if val.get("ddm_value"): block += f" | DDM: ${val['ddm_value']}" if tech: block += f""" Technicals: RSI {tech['rsi']} | MACD hist {tech['macd_hist']:+.3f} | {'Above' if tech['above_ma50'] else 'Below'} MA50 | Signal: {tech['signal']} Recent range: 
t
e
c
h
[
′
s
u
p
p
o
r
t
′
]
–
tech[ 
′
 support 
′
 ]–{tech['resistance']}""" block += f"\nRecent news:\n{news}" sections.append(block)

full_context = "\n\n".join(sections)

prompt = f"""You are a portfolio analyst writing a pre-market deep research brief for a Bangkok-based investor, 2 hours before US market open. Their full 14-stock portfolio data is below — current prices, cost basis, fundamentals, technicals, and last week's news per ticker.
{full_context}

Write a comprehensive Markdown brief in this exact structure:

Executive Summary
3-4 sentences: overall portfolio health, biggest risk for today's session, biggest opportunity for today's session.

Action Priorities (max 5, ranked)
Specific, actionable items with concrete price levels. Example:

Trim MU 20% above $760 — 2nd profit-take tranche; valuation stretched, P/E 32 vs 5y median ~18.
Add to ORCL on dip to $185 — fair value $245, thesis intact, RSI 42 bottoming.
Cite the actual numbers from the data above. No vague language.

Per-Stock Status (all 14)
Format: TICKER 🟢/🟡/🔴 — current state in 1 line + key catalyst this week + action.

Top 3 Risks Today
Portfolio-wide risks — macro, sector, or specific names. 1 line each.

Watch at Open
2-3 specific tickers and what to look for in the first 30 min of trading.

Tone: direct, specific, no filler, no disclaimers."""

return call_gemini(prompt, max_tokens=4000, model=MODEL_DEEP)
===== TELEGRAM =====
def telegram_send(text): token = os.environ["TELEGRAM_BOT_TOKEN"] chat = os.environ["TELEGRAM_CHAT_ID"] for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]: try: requests.post( f"https://api.telegram.org/bot{token}/sendMessage", data={"chat_id": chat, "text": chunk, "parse_mode": "Markdown"}, timeout=10, ) except Exception as e: print(f"tg err: {e}")

===== MAIN =====
def run_scout(): cache = load_cache() result = scout_news(cache) msgs = [] if result.get("urgent"): msg = "🚨 URGENT\n\n" for u in result["urgent"]: sev = "🔴" if u.get("severity") == "critical" else "🟠" msg += f"{sev} {u['ticker']} — {u['headline']}\n_{u['why']}_\n\n" msgs.append(msg) if result.get("candidates"): analyzed = [analyze_candidate(c, cache) for c in result["candidates"][:5]] recs = [format_recommendation(a, action) for a in analyzed if (action := decide_action(a))] if recs: msgs.append("🔍 Candidates\n\n" + "\n———\n\n".join(recs)) save_cache(cache) for m in msgs: telegram_send(m)

def run_deep(): cache = load_cache() brief = deep_research(cache) save_cache(cache) header = f"📊 Deep Research — {datetime.utcnow().strftime('%b %d %Y')}\n_2 hours to US open_\n\n" telegram_send(header + brief)

if name == "main": mode = sys.argv[1] if len(sys.argv) > 1 else "scout" if mode == "deep": run_deep() else: run_scout()

