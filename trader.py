import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
from bs4 import BeautifulSoup
import time
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import FinanceDataReader as fdr
from pykrx import stock
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
STATE_DIR = "state"
STRATEGY_STATE_PATH = os.path.join(STATE_DIR, "strategy_state.json")
ANALYSIS_CACHE_PATH = os.path.join(STATE_DIR, "analysis_cache.json")

# LLM Providers: "openai" or "gemini"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_LITE_MODEL = "gpt-4o-mini" # Prompt 1-4용
OPENAI_PRO_MODEL = "gpt-4o"      # Prompt 5-6용

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_LITE_MODEL = "gemini-2.5-flash-lite"    # Prompt 1-4용
GEMINI_PRO_MODEL = "gemini-3-pro-preview"      # Prompt 5-6용

# Global Model Assignment based on Provider
if LLM_PROVIDER == "gemini":
    LITE_MODEL = GEMINI_LITE_MODEL
    PRO_MODEL = GEMINI_PRO_MODEL
else:
    LITE_MODEL = OPENAI_LITE_MODEL
    PRO_MODEL = OPENAI_PRO_MODEL

LLM_DISABLED = os.getenv("LLM_DISABLED", "0").strip() == "1"

MAX_PORTFOLIO_STOCKS = 5
TRAJECTORY_K = 30 # 성과 추적을 위해 보관 기간 연장

SCORING_DIMENSIONS = [
    "financial_health",
    "growth_potential",
    "news_sentiment",
    "news_impact",
    "price_momentum",
    "volatility_risk"
]

# --- Helper Functions ---

def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"```$", "", text)
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match: return json.loads(match.group(1))
    raise ValueError("No JSON found")

def _llm_chat(messages: List[Dict[str, str]], model: str = None, temperature=0.2) -> str:
    if os.getenv("LLM_DISABLED", "0").strip() == "1": return "{}"
    target_model = model if model else LITE_MODEL
    if LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY: return "{}"
        prompt = "\n".join([m['content'] for m in messages])
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={GEMINI_API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": temperature}}
        res = requests.post(url, json=payload, timeout=60)
        res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        if not OPENAI_API_KEY: return "{}"
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": target_model, "messages": messages, "temperature": temperature}
        res = requests.post(url, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

def get_latest_trading_day():
    today = datetime.now().strftime("%Y%m%d")
    try:
        df = stock.get_market_ohlcv((datetime.now() - timedelta(days=10)).strftime("%Y%m%d"), today, "005930")
        if df.empty: return today
        return df.index[-1].strftime("%Y%m%d")
    except: return today

def get_stock_universe(limit=30) -> List[str]:
    """코스닥 시총 상위 limit개 종목을 가져옵니다. (흑자 필터링 제거)"""
    try:
        print(f"코스닥 시총 상위 {limit}개 종목 수집 중...")
        df_kq = fdr.StockListing('KOSDAQ')
        df_kq = df_kq.sort_values(by='Marcap', ascending=False).head(limit)
        return [f"{c}.KQ" for c in df_kq['Code'].tolist()]
    except Exception as e:
        print(f"Universe 수집 오류: {e}")
        return ["247540.KQ", "086520.KQ"]

# --- Checkpoint & Performance Systems ---

def load_cache(today_str: str) -> Dict[str, Any]:
    if os.path.exists(ANALYSIS_CACHE_PATH):
        try:
            data = json.load(open(ANALYSIS_CACHE_PATH))
            if data.get("date") == today_str: return data.get("results", {})
        except: pass
    return {}

def save_cache(today_str: str, results: Dict[str, Any]):
    os.makedirs(STATE_DIR, exist_ok=True)
    json.dump({"date": today_str, "results": results}, open(ANALYSIS_CACHE_PATH, 'w'), ensure_ascii=False, indent=2)

def _normalize_scores(raw_scores: Dict[str, Any]) -> Dict[str, int]:
    """LLM 응답 점수를 정규화합니다 (NaN 방지)."""
    normalized = {d: 5 for d in SCORING_DIMENSIONS}
    mapping = {
        "financial_health": ["financial_health", "financial", "profitability", "valuation", "fundamental_strength"],
        "growth_potential": ["growth_potential", "growth", "potential", "growth_prospects"],
        "news_sentiment": ["news_sentiment", "sentiment", "market_sentiment", "overall_tone"],
        "news_impact": ["news_impact", "impact", "influence", "news_influence"],
        "price_momentum": ["price_momentum", "momentum", "technical", "trend"],
        "volatility_risk": ["volatility_risk", "volatility", "risk", "stability", "risk_level"]
    }
    for target, syns in mapping.items():
        for s in syns:
            if s in raw_scores:
                try: 
                    val = raw_scores[s]
                    if isinstance(val, (int, float)):
                        normalized[target] = int(val)
                        break
                except: pass
    return normalized

def calculate_performance(trajectory: List[Dict]) -> List[Dict]:
    """과거 추천 종목들의 현재 수익률을 실시간 업데이트합니다."""
    if not trajectory: return []
    print("과거 추천주 실시간 성과 추적 중...")
    all_tickers = set()
    for entry in trajectory:
        selected = entry.get("selected", [])
        if isinstance(selected, list):
            for s in selected:
                code = s.get("stock_code") if isinstance(s, dict) else str(s)
                match = re.search(r'(\d{6}\.K[SQ])', str(code))
                if match: all_tickers.add(match.group(1))
                else: all_tickers.add(str(code))
    
    if not all_tickers: return trajectory

    current_prices = {}
    try:
        data = yf.download(list(all_tickers), period="1d", progress=False)
        for t in all_tickers:
            try:
                price_val = data['Close'][t].iloc[-1]
                current_prices[t] = float(price_val.iloc[0]) if hasattr(price_val, 'iloc') else float(price_val)
            except: continue
    except: pass

    for entry in trajectory:
        total_ret = 0.0; total_weight = 0.0
        selected = entry.get("selected", [])
        if not isinstance(selected, list): continue
        for s in selected:
            if not isinstance(s, dict): continue
            raw_code = s.get("stock_code")
            match = re.search(r'(\d{6}\.K[SQ])', str(raw_code))
            code = match.group(1) if match else str(raw_code)
            buy_price = s.get("buy_price"); weight = s.get("weight", 1)
            if code in current_prices and buy_price and buy_price > 0:
                ret = ((current_prices[code] / buy_price) - 1) * 100
                s["current_price"] = int(current_prices[code]); s["return"] = round(ret, 2)
                total_ret += ret * weight; total_weight += weight
        entry["perf"] = round(total_ret / total_weight, 2) if total_weight > 0 else 0.0
    return trajectory

# --- Core Data Fetcher ---

def _get_stock_data(ticker: str) -> Dict[str, Any]:
    code = ticker.split(".")[0]
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty: return {}
        def safe_get(series, idx=-1):
            val = series.iloc[idx]
            return float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
        last_close = safe_get(df['Close'])
        tech_summary = f"Price: {int(last_close)}, Weekly: {round(((last_close / safe_get(df['Close'], -6)) - 1) * 100, 2)}%"
        
        soup_main = BeautifulSoup(requests.get(f"https://finance.naver.com/item/main.naver?code={code}", headers={"User-Agent":"Mozilla/5.0"}).text, "html.parser")
        def _parse(s, i):
            try: return float(s.find("em", id=i).text.replace(",","").replace("배","").replace("%",""))
            except: return 0.0
        fund_data = {"per": _parse(soup_main, "_per"), "pbr": _parse(soup_main, "_pbr")}
        
        news_res = requests.get(f"https://m.stock.naver.com/api/news/stock/{code}?pageSize=5&page=1", headers={"User-Agent":"Mozilla/5.0"}).json()
        news_contexts = [f"[{i.get('title','')}] {i.get('body','')}".replace('&quot;','"') for e in news_res if 'items' in e for i in e['items']]
        
        return {"tech_text": tech_summary, "fund_text": str(fund_data), "news_text": "\n".join(news_contexts), "price": int(last_close)}
    except: return {}

# --- Multi-Agent Pipeline (Exact Paper Prompts) ---

def news_agent(ticker: str, raw_news: str) -> str:
    """Prompt 1: News Agent (Paper Version)"""
    prompt = f"""You are a financial news analysis agent. Your task is to filter and summarize recent news related to the stock {ticker}. The news content below includes summaries or full articles from the past week:
{raw_news}

Please provide a concise and insightful weekly summary of the stock's recent news. Your output will be used to help a downstream stock selection agent make informed weekly investment decisions."""
    return _llm_chat([{"role": "user", "content": prompt}], model=LITE_MODEL)

def technical_agent(ticker: str, tech_text: str) -> str:
    """Prompt 2: Technical Agent (Paper Version)"""
    prompt = f"""You are a stock price analysis agent. Your task is to analyze the recent technical indicators and price data of the stock {ticker}. Below is the stock's daily technical indicator and prices from the past 4 weeks:
{tech_text}.

Please provide a technical analysis of the stock's recent performance. Your output will be used to help a downstream stock selection agent make informed weekly investment decisions."""
    return _llm_chat([{"role": "user", "content": prompt}], model=LITE_MODEL)

def fundamental_agent(ticker: str, fund_text: str) -> str:
    """Prompt 3: Fundamental Agent (Paper Version)"""
    prompt = f"""You are a stock fundamentals analysis agent. Your task is to analyze the recent financial performance of the stock {ticker} based on its past 4 quarterly reports. Below is the stock's recent financial data, including 4 quarters of: Income statements, Balance sheets, Cash flow statements.
{fund_text}.

Please provide a summary of the stock's fundamental trends. You may consider trends in revenue, profit, expenses, margins, cash flow, and balance sheet strength, as well as any notable improvements or warning signs."""
    return _llm_chat([{"role": "user", "content": prompt}], model=LITE_MODEL)

def score_agent(ticker: str, news_summ: str, fund_summ: str, tech_anal: str) -> Dict[str, Any]:
    """Prompt 4: Score Agent (Paper Version)"""
    prompt = f"""You are an expert stock evaluation assistant. Tasked with assessing each stock using three input types: News summary, Fundamental analysis, and Recent price behavior.

From these inputs, evaluate the stock along six scoring dimensions. For each dimension: provide a score from 1 to 10, and give a brief justification (1-2 short sentences max).

Use only the information provided below. If anything is missing, score conservatively and state that in the reason.

**stock**: {ticker}
**News Summary**: {news_summ}
**Fundamental Analysis**: {fund_summ}
**Price and Technical Analysis**: {tech_anal}

**Scoring Dimensions** (1-10):
1. financial_health – based on profitability, debt, cash flow, etc.
2. growth_potential – based on investment plans, innovation, and expansion prospects.
3. news_sentiment – overall tone of the news.
4. news_impact – the breadth and duration of news influence.
5. price_momentum – recent trends, strength, and consistency.
6. volatility_risk – recent price stability or instability (higher score = more STABLE/LESS risk)

Return ONLY JSON format:
{{
  "scores": {{ "financial_health": X, "growth_potential": X, "news_sentiment": X, "news_impact": X, "price_momentum": X, "volatility_risk": X }},
  "justifications": {{ ... }}
}}"""
    try:
        res = _extract_json(_llm_chat([{"role": "user", "content": prompt}], model=LITE_MODEL))
        res['scores'] = _normalize_scores(res.get('scores', {}))
        return res
    except: return {"scores": {d: 5 for d in SCORING_DIMENSIONS}}

def strategy_agent(trajectory: List[Dict], market_overview: str) -> str:
    """Prompt 6: Strategy Agent (Paper Version)"""
    history_text = json.dumps(trajectory[-5:], ensure_ascii=False, indent=2)
    prompt = f"""You are a strategic investment advisor tasked with refining portfolio strategy based on historical performance and current market signals. Your inputs include:

1. Recent Strategy History:
{history_text}
(A list of previous portfolio strategies, their observed returns, and the average return of the candidate stock universe.)

2. Current Market Signals:
{market_overview}

Your task is to analyze the past performance of different strategies and provide a **refined, data-driven strategy recommendation** for the upcoming week.
Please provide a concise and professional strategy description."""
    return _llm_chat([{"role": "user", "content": prompt}], model=PRO_MODEL, temperature=0.5)

def selection_agent(strategy: str, candidates: List[Dict]) -> Dict[str, Any]:
    """Prompt 5: Selector Agent (Paper Version)"""
    scoring_reports = "\n".join([f"- {c['name']} ({c['ticker']}): {c['scores']}" for c in candidates])
    prompt = f"""As an experienced stock-picking expert, your task is to construct a prudent and strategically aligned portfolio for the next holding period. You are provided with two sources of information:

1. Score reports for various stocks:
{scoring_reports}

2. A recommended strategy for the upcoming period:
{strategy}

Using these inputs, select the most suitable stocks that align well with the recommended strategy. Choose **up to 5 stocks**. Return ONLY JSON with 'selected_stocks' (list of {{stock_code, weight}}) and 'reasoning'."""
    try: return _extract_json(_llm_chat([{"role": "user", "content": prompt}], model=PRO_MODEL))
    except: return {"selected_stocks": [{"stock_code": c['ticker'], "weight": 20} for c in candidates[:5]]}

def get_market_overview() -> str:
    try:
        end = get_latest_trading_day(); start = (datetime.strptime(end, "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
        df = stock.get_market_ohlcv(start, end, "101", market="KOSDAQ")
        news = [t.text.strip() for t in BeautifulSoup(requests.get("https://finance.naver.com/news/mainnews.naver", headers={"User-Agent":"Mozilla/5.0"}).text, "html.parser").select(".mainnews_list .articleSubject a")[:3]]
        return f"KOSDAQ Index: {df['종가'].iloc[-1]}. News: {', '.join(news)}"
    except: return "Stable market."

def main():
    print("3S-Trader KR: High-Fidelity Multi-Agent Mode (Clean Universe)")
    today_str = datetime.now().strftime('%Y-%m-%d')
    cache = load_cache(today_str)
    
    market_overview = get_market_overview()
    trajectory = []
    if os.path.exists(STRATEGY_STATE_PATH):
        try: trajectory = json.load(open(STRATEGY_STATE_PATH)).get("trajectory", [])
        except: pass
    current_strategy = strategy_agent(trajectory, market_overview)
    print(f"Strategy: {current_strategy[:50]}...")

    universe = get_stock_universe(limit=30)
    to_do = [t for t in universe if t not in cache]
    
    def process(t):
        raw = _get_stock_data(t)
        if not raw: return None
        n, te, f = news_agent(t, raw['news_text']), technical_agent(t, raw['tech_text']), fundamental_agent(t, raw['fund_text'])
        res = score_agent(t, n, f, te)
        print(f"[Anal: {t}] Scores: {res.get('scores', {})}")
        return t, {"ticker": t, "name": stock.get_market_ticker_name(t.split('.')[0]), "scores": res.get('scores', {}), "data": {"price": raw['price']}}

    if to_do:
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(process, t) for t in to_do]
            for fut in as_completed(futures):
                r = fut.result(); 
                if r: cache[r[0]] = r[1]; save_cache(today_str, cache)

    scored_universe = [cache[t] for t in universe if t in cache]
    scored_sorted = sorted(scored_universe, key=lambda x: sum(x['scores'].values()), reverse=True)
    sel_res = selection_agent(current_strategy, scored_sorted[:30])
    final_stocks = sel_res.get("selected_stocks", [])
    
    price_map = {s['ticker']: s['data']['price'] for s in scored_universe}
    for s in final_stocks: s['buy_price'] = price_map.get(str(s.get('stock_code','')), 0)
    
    today_entry = {"date": today_str, "strategy": current_strategy, "selected": final_stocks, "perf": 0.0}
    found_idx = next((i for i, e in enumerate(trajectory) if e.get("date") == today_str), -1)
    if found_idx >= 0: trajectory[found_idx] = today_entry
    else: trajectory.append(today_entry)
    
    trajectory = calculate_performance(trajectory)
    json.dump({"trajectory": trajectory[-TRAJECTORY_K:]}, open(STRATEGY_STATE_PATH, 'w'), ensure_ascii=False, indent=2)

    filename = f"reports/3S_Trader_Report_{today_str}.md"; os.makedirs("reports", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 3S-Trader KR 전략 리포트 ({today_str})\n\n## 🧠 1. Strategy\n{current_strategy}\n\n")
        f.write("## 📈 2. Performance Tracking (과거 추천 성과)\n")
        past_entries = [e for e in trajectory if e['date'] != today_str]
        if past_entries:
            perf_list = []
            for t_entry in reversed(past_entries):
                picks = []
                for s in t_entry.get('selected', []):
                    raw_c = s.get('stock_code', 'N/A')
                    match = re.search(r'(\d{6}\.K[SQ])', str(raw_c))
                    clean_c = match.group(1) if match else str(raw_c)
                    picks.append(f"{clean_c} ({s.get('return', 0)}%)")
                perf_list.append({"추천일": t_entry['date'], "추천종목 (수익률)": ", ".join(picks[:5]), "평균수익률": f"{t_entry.get('perf', 0)}%"})
            f.write(pd.DataFrame(perf_list).head(10).to_markdown(index=False) + "\n\n")
        else: f.write("*과거 기록이 없습니다.*\n\n")

        f.write(f"## 🎯 3. Selection (Today's TOP 5)\n")
        selected_tickers = [str(s.get('stock_code','')) for s in final_stocks]
        selected_data = []
        for s in final_stocks:
            code = str(s.get('stock_code',''))
            match = next((x for x in scored_universe if code in x['ticker'] or x['ticker'] in code), None)
            if match: selected_data.append((match, s.get('weight', 0)))

        if selected_data:
            df = pd.DataFrame([{"종목명": m['name'], "티커": m['ticker'], "비중": w, "현재가": m['data']['price'], "Total": sum(m['scores'].values())} for m, w in selected_data])
            f.write(df.sort_values("비중", ascending=False).to_markdown(index=False) + "\n\n")
        else: f.write("*선택 실패*\n\n")
        
        f.write("## 📊 4. Scoring Detail\n")
        f.write(pd.DataFrame([{"종목명": s['name'], "티커": s['ticker'], **s['scores']} for s in scored_universe]).to_markdown(index=False))
    print(f"Report: {filename}")

if __name__ == "__main__": main()
