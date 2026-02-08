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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o").strip()
LLM_DISABLED = os.getenv("LLM_DISABLED", "0").strip() == "1"

MAX_PORTFOLIO_STOCKS = 5
TRAJECTORY_K = 10 

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
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise ValueError("No JSON found in LLM response")

def _openai_chat(messages: List[Dict[str, str]], temperature=0.2) -> str:
    if not OPENAI_API_KEY: return "{}"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "messages": messages, "temperature": temperature}
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

def is_profitable(code: str) -> bool:
    try:
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        table = soup.find("table", class_="tb_type1 tb_num")
        if not table: return True
        rows = table.find_all("tr")
        op_row = next((r for r in rows if "영업이익" in r.text), None)
        if not op_row: return True
        cols = op_row.find_all("td")
        for col in reversed(cols):
            val_str = col.text.strip().replace(",", "")
            if val_str and val_str != "-":
                try: return float(val_str) > 0
                except: continue
        return True
    except: return True

def get_stock_universe() -> List[str]:
    try:
        print("코스닥 시총 상위 100개 종목 수집 중...")
        df_kq = fdr.StockListing('KOSDAQ')
        df_kq = df_kq.sort_values(by='Marcap', ascending=False).head(100)
        top_codes = df_kq['Code'].tolist()
        
        print(f"흑자 필터링 시작 (대상: {len(top_codes)} 종목)...")
        profitable_universe = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_code = {executor.submit(is_profitable, c): c for c in top_codes}
            for future in as_completed(future_to_code):
                if future.result(): profitable_universe.append(f"{future_to_code[future]}.KQ")
        print(f"\n최종 Universe 구성 완료: {len(profitable_universe)} 종목")
        return profitable_universe
    except Exception as e:
        print(f"Universe 수집 오류: {e}")
        return ["247540.KQ", "086520.KQ"]

# --- Core Data Fetcher ---

def _get_stock_data(ticker: str) -> Dict[str, Any]:
    code = ticker.split(".")[0]
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty: return {}
        
        # Series indexing fix for FutureWarning
        last_close_val = df['Close'].iloc[-1]
        last_close = float(last_close_val.iloc[0]) if hasattr(last_close_val, 'iloc') else float(last_close_val)
        
        prev_close_val = df['Close'].iloc[-6]
        prev_close = float(prev_close_val.iloc[0]) if hasattr(prev_close_val, 'iloc') else float(prev_close_val)
        
        weekly_return = round(((last_close / prev_close) - 1) * 100, 2)
        vol_series = df['Close'].pct_change().tail(20).std()
        volatility = round(float(vol_series.iloc[0] if hasattr(vol_series, 'iloc') else vol_series) * 100, 2)
        
        ma5_series = df['Close'].rolling(window=5).mean().iloc[-1]
        ma5 = float(ma5_series.iloc[0] if hasattr(ma5_series, 'iloc') else ma5_series)
        
        ma20_series = df['Close'].rolling(window=20).mean().iloc[-1]
        ma20 = float(ma20_series.iloc[0] if hasattr(ma20_series, 'iloc') else ma20_series)
        
        ma60_series = df['Close'].rolling(window=60).mean().iloc[-1]
        ma60 = float(ma60_series.iloc[0] if hasattr(ma60_series, 'iloc') else ma60_series)
        
        gap5 = round(((last_close / ma5) - 1) * 100, 2)
        gap20 = round(((last_close / ma20) - 1) * 100, 2)
        
        mom1m_val = df['Close'].iloc[-20] if len(df) >= 20 else df['Close'].iloc[0]
        mom1m_price = float(mom1m_val.iloc[0] if hasattr(mom1m_val, 'iloc') else mom1m_val)
        mom1m = round(((last_close / mom1m_price) - 1) * 100, 2)
        
        mom3m_val = df['Close'].iloc[-60] if len(df) >= 60 else df['Close'].iloc[0]
        mom3m_price = float(mom3m_val.iloc[0] if hasattr(mom3m_val, 'iloc') else mom3m_val)
        mom3m = round(((last_close / mom3m_price) - 1) * 100, 2)
        
        delta = df['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(window=14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_val = rs.iloc[-1]
        last_rs = float(rsi_val.iloc[0] if hasattr(rsi_val, 'iloc') else rsi_val)
        rsi = round(100 - (100 / (1 + last_rs)), 2)
        
        tech_data = {"price": int(last_close), "weekly_return": weekly_return, "volatility": volatility, "ma_status": "정배열" if ma5 > ma20 > ma60 else "역배열/혼조", "ma_gaps": {"ma5": gap5, "ma20": gap20}, "momentum": {"1m": mom1m, "3m": mom3m}, "rsi": rsi}
        
        # Fundamental & Investor (Naver)
        url_main = f"https://finance.naver.com/item/main.naver?code={code}"
        soup_main = BeautifulSoup(requests.get(url_main, headers={"User-Agent": "Mozilla/5.0"}).text, "html.parser")
        def _parse_naver_val(soup, id_str):
            try: return float(soup.find("em", id=id_str).text.replace(",", "").replace("배", "").replace("%", ""))
            except: return 0.0
        fundamental = {"per": _parse_naver_val(soup_main, "_per"), "pbr": _parse_naver_val(soup_main, "_pbr"), "div_yield": _parse_naver_val(soup_main, "_dvr")}
        
        url_frgn = f"https://finance.naver.com/item/frgn.naver?code={code}"
        soup_frgn = BeautifulSoup(requests.get(url_frgn, headers={"User-Agent": "Mozilla/5.0"}).text, "html.parser")
        rows = soup_frgn.select("table.type2 tr"); f_sum, i_sum, count = 0, 0, 0
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 9:
                try: f_sum += int(cols[6].text.replace(",", "")); i_sum += int(cols[5].text.replace(",", "")); count += 1
                except: continue
            if count >= 5: break
        investor = {"foreign": f_sum, "institution": i_sum}
        
        news_contexts = []
        res_news = requests.get(f"https://m.stock.naver.com/api/news/stock/{code}?pageSize=5&page=1", headers={"User-Agent": "Mozilla/5.0"}).json()
        for entry in res_news:
            if 'items' in entry:
                for item in entry['items']:
                    news_contexts.append(f"[{item.get('title', '')}] {item.get('body', '')}".replace('&quot;', '"'))
        return {**tech_data, **fundamental, **investor, "news_contexts": news_contexts}
    except Exception as e: return {}

# --- Efficient Multi-Agent Analysis ---

def analyze_stock_unified(ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """논문의 4가지 에이전트 역할을 하나의 고밀도 프롬프트로 통합 (효율 극대화)"""
    if LLM_DISABLED or not data: return {"scores": {d: 5 for d in SCORING_DIMENSIONS}}
    
    prompt = f"""
Act as a panel of experts for stock {ticker}. Follow these steps in one internal reasoning process:

1. [News Agent]: Summarize these: {data.get('news_contexts', [])}
2. [Technical Agent]: Analyze: Price {data['price']}, Weekly {data['weekly_return']}%, 1m/3m Momentum {data['momentum']}, RSI {data['rsi']}, MA Gaps {data['ma_gaps']}
3. [Fundamental Agent]: Analyze: PER {data['per']}, PBR {data['pbr']}, DivYield {data['div_yield']}%, InvestorNet {data['foreign']}/{data['institution']}
4. [Score Agent (Prompt4)]: Evaluate along six dimensions (1-10). Provide 1-2 sentence justification for each.
   Dimensions: Financial Health, Growth Potential, News Sentiment, News Impact, Price Momentum, Volatility Risk (High score = low risk).

Return ONLY JSON:
{{
  "summaries": {{ "news": "...", "technical": "...", "fundamental": "..." }},
  "scores": {{ "financial_health": 5, "growth_potential": 5, "news_sentiment": 5, "news_impact": 5, "price_momentum": 5, "volatility_risk": 5 }},
  "justifications": {{ "financial_health": "...", ... }}
}}
"""
    try: return _extract_json(_openai_chat([{"role": "user", "content": prompt}]))
    except: return {"scores": {d: 5 for d in SCORING_DIMENSIONS}}

def strategy_agent(trajectory: List[Dict], market_overview: str) -> str:
    prompt = f"Market: {market_overview}\nHistory: {trajectory}\nTask: Define selection strategy. Prioritize dimensions. Return concise text."
    try: return _openai_chat([{"role": "user", "content": prompt}], temperature=0.5)
    except: return "Focus on momentum and value."

def selection_agent(strategy: str, candidates: List[Dict]) -> List[str]:
    prompt = f"Strategy: {strategy}\nCandidates: {candidates}\nTask: Select top {MAX_PORTFOLIO_STOCKS}. Return JSON list of tickers."
    try: return _extract_json(_openai_chat([{"role": "user", "content": prompt}]))
    except: return [c['ticker'] for c in candidates[:MAX_PORTFOLIO_STOCKS]]

def get_market_overview() -> str:
    try:
        end = get_latest_trading_day(); start = (datetime.strptime(end, "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
        df = stock.get_market_ohlcv(start, end, "101", market="KOSDAQ")
        curr = df['종가'].iloc[-1]; change = round(((curr / df['종가'].iloc[-2]) - 1) * 100, 2)
        news = [t.text.strip() for t in BeautifulSoup(requests.get("https://finance.naver.com/news/mainnews.naver", headers={"User-Agent": "Mozilla/5.0"}).text, "html.parser").select(".mainnews_list .articleSubject a")[:3]]
        return f"KOSDAQ: {curr} ({change}%). News: {', '.join(news)}"
    except: return "Market stable."

def main():
    print("3S-Trader KR: Efficient Multi-Agent Mode Starting...")
    if not os.path.exists(STATE_DIR): os.makedirs(STATE_DIR)
    trajectory = []
    if os.path.exists(STRATEGY_STATE_PATH):
        try: trajectory = json.load(open(STRATEGY_STATE_PATH)).get("trajectory", [])
        except: pass

    market_overview = get_market_overview()
    current_strategy = strategy_agent(trajectory, market_overview)
    print(f"Strategy: {current_strategy[:50]}...")

    universe_tickers = get_stock_universe()
    scored_universe = []
    
    def process_stock(t):
        data = _get_stock_data(t)
        if not data: return None
        analysis = analyze_stock_unified(t, data)
        return {"ticker": t, "name": stock.get_market_ticker_name(t.split('.')[0]), "scores": analysis.get("scores", {}), "data": data}

    print("Analyzing stocks in parallel...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_stock, t) for t in universe_tickers]
        for future in as_completed(futures):
            res = future.result()
            if res: scored_universe.append(res)
    
    scored_sorted = sorted(scored_universe, key=lambda x: sum(x['scores'].values()), reverse=True)
    final_tickers = selection_agent(current_strategy, scored_sorted[:30])
    
    trajectory.append({"date": datetime.now().strftime("%Y-%m-%d"), "strategy": current_strategy, "selected": final_tickers, "perf": 0.0})
    json.dump({"trajectory": trajectory[-TRAJECTORY_K:]}, open(STRATEGY_STATE_PATH, 'w'))

    today_str = datetime.now().strftime('%Y-%m-%d'); filename = f"reports/3S_Trader_Report_{today_str}.md"; os.makedirs("reports", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 3S-Trader KR 전략 리포트 ({today_str})\n\n## 🧠 1. Strategy\n{current_strategy}\n\n## 🎯 2. Selection\n")
        selected_data = [s for s in scored_universe if s['ticker'] in final_tickers]
        if selected_data: f.write(pd.DataFrame([{"종목명": s['name'], "티커": s['ticker'], "현재가": s['data']['price'], "Total점수": sum(s['scores'].values())} for s in selected_data]).to_markdown(index=False))
        f.write("\n\n## 📊 3. Scoring Detail\n")
        f.write(pd.DataFrame([{"종목명": s['name'], "티커": s['ticker'], **s['scores']} for s in scored_universe]).to_markdown(index=False))
    print(f"Report: {filename}")

if __name__ == "__main__": main()
