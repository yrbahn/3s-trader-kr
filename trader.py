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

# --- Configuration ---
STATE_DIR = "state"
STRATEGY_STATE_PATH = os.path.join(STATE_DIR, "strategy_state.json")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o").strip()
LLM_DISABLED = os.getenv("LLM_DISABLED", "0").strip() == "1"

MAX_PORTFOLIO_STOCKS = 5
TRAJECTORY_K = 10 # 과거 궤적 참조 개수

SCORING_DIMENSIONS = [
    "financial_health",
    "growth_potential",
    "news_sentiment",
    "news_impact",
    "price_momentum",
    "volatility_risk"
]

# --- Helper Functions ---

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        if isinstance(x, (int, float, np.number)): return float(x)
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() in {"nan", "none"}: return None
        return float(s)
    except Exception: return None

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
    """가장 최근 영업일을 구합니다."""
    today = datetime.now().strftime("%Y%m%d")
    try:
        # 삼성전자 데이터를 통해 실제 데이터가 있는 영업일을 확인
        df = stock.get_market_ohlcv((datetime.now() - timedelta(days=10)).strftime("%Y%m%d"), today, "005930")
        if df.empty: return today
        df = df[df['종가'] > 0]
        return df.index[-1].strftime("%Y%m%d")
    except:
        return today

def get_stock_universe() -> List[str]:
    """FinanceDataReader를 사용하여 코스닥 상위 종목 Universe를 구성합니다."""
    fallback_tickers = [
        '247540', '086520', '191170', '028300', '291230', 
        '068760', '403870', '058470', '272410', '214150'
    ]
    try:
        df_kq = fdr.StockListing('KOSDAQ')
        if df_kq.empty:
            return [f"{t}.KQ" for t in fallback_tickers]
        
        cap_col = next((c for c in ['Marcap', '시가총액', 'Amount'] if c in df_kq.columns), None)
        if cap_col:
            df_kq = df_kq.sort_values(by=cap_col, ascending=False)
        
        code_col = next((c for c in ['Code', 'Symbol', '종목코드'] if c in df_kq.columns), 'Code')
        top_tickers = df_kq[code_col].head(15).tolist()
        return [f"{t}.KQ" for t in top_tickers]
    except Exception as e:
        print(f"Universe 수집 오류 (로컬 리스트 사용): {e}")
        return [f"{t}.KQ" for t in fallback_tickers]

# --- 1. Scoring Module ---

def _get_stock_data(ticker: str) -> Dict[str, Any]:
    """Scoring을 위한 원천 데이터 수집 (Technical + News)"""
    try:
        data = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if data.empty: return {}
        current_price = data['Close'].iloc[-1]
        weekly_return = (data['Close'].iloc[-1] / data['Close'].iloc[-6] - 1) * 100
        volatility = data['Close'].pct_change().tail(20).std() * 100
        
        code = ticker.split(".")[0]
        url = f"https://finance.naver.com/item/news_news.naver?code={code}"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        headlines = [a.text.strip() for a in soup.select("table.type5 a")[:5]]
        
        return {
            "price": int(current_price),
            "weekly_return": round(weekly_return, 2),
            "volatility": round(volatility, 2),
            "headlines": headlines
        }
    except: return {}

def scoring_agent(ticker: str, data: Dict[str, Any]) -> Dict[str, int]:
    """LLM이 데이터를 보고 6개 차원에 대해 점수 산출 (1-10)"""
    if LLM_DISABLED or not data:
        return {d: 5 for d in SCORING_DIMENSIONS}
    prompt = f"""Analyze the stock {ticker} based on following data:
- Weekly Return: {data['weekly_return']}%
- 20-day Volatility: {data['volatility']}%
- Recent Headlines: {data['headlines']}
Assign scores (1-10) for each dimension: {SCORING_DIMENSIONS}
Return ONLY JSON format: {{"scores": {{"dim_name": score, ...}}, "rationale": "short string"}}"""
    try:
        res = _openai_chat([{"role": "user", "content": prompt}])
        return _extract_json(res).get("scores", {d: 5 for d in SCORING_DIMENSIONS})
    except: return {d: 5 for d in SCORING_DIMENSIONS}

# --- 2. Strategy Module ---

def strategy_agent(trajectory: List[Dict], market_overview: str) -> str:
    if LLM_DISABLED: return "Select stocks with high momentum."
    prompt = f"""Current Market: {market_overview}\nPast Trajectory: {trajectory}\nTask: define a selection strategy. Return concise description."""
    try: return _openai_chat([{"role": "user", "content": prompt}], temperature=0.5)
    except: return "Focus on high momentum and positive news sentiment."

# --- 3. Selection Module ---

def selection_agent(strategy: str, scored_universe: List[Dict]) -> List[str]:
    if LLM_DISABLED:
        return [s['ticker'] for s in sorted(scored_universe, key=lambda x: sum(x['scores'].values()), reverse=True)[:MAX_PORTFOLIO_STOCKS]]
    prompt = f"""Strategy: {strategy}\nScored Stocks: {scored_universe[:15]}\nTask: Select best {MAX_PORTFOLIO_STOCKS} stocks. Return JSON list: ["code1.KQ", ...]"""
    try:
        res = _openai_chat([{"role": "user", "content": prompt}])
        return _extract_json(res)
    except:
        return [s['ticker'] for s in sorted(scored_universe, key=lambda x: sum(x['scores'].values()), reverse=True)[:MAX_PORTFOLIO_STOCKS]]

# --- Main Execution ---

def main():
    print("3S-Trader KR: Multi-LLM Framework Implementation Starting...")
    if not os.path.exists(STATE_DIR): os.makedirs(STATE_DIR)
    
    # 0. Load State
    trajectory = []
    if os.path.exists(STRATEGY_STATE_PATH):
        try:
            with open(STRATEGY_STATE_PATH, 'r') as f:
                trajectory = json.load(f).get("trajectory", [])
        except: pass

    # 1. Strategy Generation
    market_date = get_latest_trading_day()
    market_overview = "Market base date: " + market_date
    current_strategy = strategy_agent(trajectory, market_overview)
    print(f"Strategy 수립 완료: {current_strategy[:50]}...")

    # 2. Scoring Universe
    universe_tickers = get_stock_universe()
    scored_universe = []
    for ticker in universe_tickers:
        print(f"Scoring {ticker}...", end='\r')
        data = _get_stock_data(ticker)
        if not data: continue
        scores = scoring_agent(ticker, data)
        scored_universe.append({
            "ticker": ticker,
            "name": stock.get_market_ticker_name(ticker.split('.')[0]),
            "scores": scores,
            "data": data
        })
    
    # 3. Selection
    final_tickers = selection_agent(current_strategy, scored_universe)
    print(f"\n최종 Selection 완료: {final_tickers}")

    # 4. Save State & Report
    trajectory.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "strategy": current_strategy,
        "selected": final_tickers,
        "perf": 0.0
    })
    with open(STRATEGY_STATE_PATH, 'w') as f:
        json.dump({"trajectory": trajectory[-TRAJECTORY_K:]}, f)

    today_str = datetime.now().strftime('%Y-%m-%d')
    filename = f"reports/3S_Trader_Report_{today_str}.md"
    os.makedirs("reports", exist_ok=True)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 3S-Trader KR 전략 리포트 ({today_str})\n\n")
        f.write(f"## 🧠 1. Strategy\n{current_strategy}\n\n")
        f.write(f"## 🎯 2. Selection\n")
        selected_data = [s for s in scored_universe if s['ticker'] in final_tickers]
        if selected_data:
            f.write(pd.DataFrame([{
                "종목명": s['name'], "티커": s['ticker'], "현재가": s['data']['price'], "Total점수": sum(s['scores'].values())
            } for s in selected_data]).to_markdown(index=False))
        f.write("\n\n## 📊 3. Scoring Detail\n")
        f.write(pd.DataFrame([{
            "종목명": s['name'], "티커": s['ticker'], **s['scores']
        } for s in scored_universe]).to_markdown(index=False))

    print(f"리포트 생성 완료: {filename}")

if __name__ == "__main__":
    main()
