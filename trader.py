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

# --- Configuration ---
STOCK_UNIVERSE = [] # main에서 시총 상위로 채워짐

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

# --- 1. Scoring Module ---

def _get_stock_data(ticker: str) -> Dict[str, Any]:
    """Scoring을 위한 원천 데이터 수집 (Technical + News)"""
    try:
        # Technical
        data = yf.download(ticker, period="6mo", interval="1d", progress=False)
        current_price = data['Close'].iloc[-1]
        weekly_return = (data['Close'].iloc[-1] / data['Close'].iloc[-6] - 1) * 100
        volatility = data['Close'].pct_change().tail(20).std() * 100
        
        # News (Naver)
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

Assign scores (1-10) for each dimension:
{SCORING_DIMENSIONS}

Return ONLY JSON format:
{{"scores": {{"dim_name": score, ...}}, "rationale": "short string"}}"""

    try:
        res = _openai_chat([{"role": "user", "content": prompt}])
        return _extract_json(res).get("scores", {d: 5 for d in SCORING_DIMENSIONS})
    except:
        return {d: 5 for d in SCORING_DIMENSIONS}

# --- 2. Strategy Module ---

def strategy_agent(trajectory: List[Dict], market_overview: str) -> str:
    """과거 궤적과 시장상황을 보고 이번 회차의 'Selection Strategy'를 텍스트로 생성"""
    if LLM_DISABLED: return "Select stocks with high momentum."
    
    prompt = f"""Current Market: {market_overview}
Past Trajectory (Performance): {trajectory}

Task: Based on the past performance and current market, define a specific strategy for stock selection. 
Focus on which dimensions (from {SCORING_DIMENSIONS}) should be prioritized.
Return a concise strategy description."""

    try:
        return _openai_chat([{"role": "user", "content": prompt}], temperature=0.5)
    except:
        return "Focus on high momentum and positive news sentiment."

# --- 3. Selection Module ---

def selection_agent(strategy: str, scored_universe: List[Dict]) -> List[str]:
    """생성된 전략에 따라 점수가 매겨진 Universe에서 최종 포트폴리오(Top 5) 선택"""
    if LLM_DISABLED:
        return [s['ticker'] for s in sorted(scored_universe, key=lambda x: sum(x['scores'].values()), reverse=True)[:5]]

    prompt = f"""Strategy: {strategy}
Scored Stocks: {scored_universe[:15]} # Top 15 for context

Task: Select the best {MAX_PORTFOLIO_STOCKS} stocks that strictly follow the given strategy.
Return ONLY JSON list of tickers: ["code1.KQ", "code2.KQ", ...]"""

    try:
        res = _openai_chat([{"role": "user", "content": prompt}])
        return _extract_json(res)
    except:
        return [s['ticker'] for s in sorted(scored_universe, key=lambda x: sum(x['scores'].values()), reverse=True)[:MAX_PORTFOLIO_STOCKS]]

# --- Main Execution ---

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

def main():
    print("3S-Trader KR: Multi-LLM Framework Implementation Starting...")
    
    # 0. Load State (Trajectory)
    if not os.path.exists(STRATEGY_STATE_PATH):
        trajectory = []
    else:
        with open(STRATEGY_STATE_PATH, 'r') as f:
            trajectory = json.load(f).get("trajectory", [])

    # 1. Strategy Generation (S)
    from pykrx import stock
    market_date = get_latest_trading_day()
    market_overview = "Market seems slightly bearish with high volatility in tech sector."
    current_strategy = strategy_agent(trajectory, market_overview)
    print(f"Strategy 수립 완료: {current_strategy[:100]}...")

    # 2. Scoring Universe (S)
    # pykrx의 get_market_ticker_list가 빈 값을 반환하는 경우가 있어, 
    # 하드코딩된 핵심 종목 리스트를 우선적으로 사용하도록 로직을 강화했습니다.
    kosdaq_top_30 = [
        '247540', '086520', '191170', '028300', '291230', 
        '068760', '403870', '058470', '272410', '214150',
        '145020', '066970', '121600', '213420', '293490'
    ]
    
    try:
        # 날짜를 지정하지 않는 것이 최신 데이터를 가져오는 데 더 안정적입니다.
        kq_tickers = stock.get_market_ticker_list(market="KOSDAQ")
        if kq_tickers:
            universe_tickers = [f"{t}.KQ" for t in kq_tickers[:15]]
        else:
            universe_tickers = [f"{t}.KQ" for t in kosdaq_top_30[:15]]
    except:
        universe_tickers = [f"{t}.KQ" for t in kosdaq_top_30[:15]]
    
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
    
    # 3. Selection (S)
    final_tickers = selection_agent(current_strategy, scored_universe)
    print(f"\n최종 Selection 완료: {final_tickers}")

    # 4. Save State & Report
    # (간이용으로 당일 수익률 0으로 기록)
    trajectory.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "strategy": current_strategy,
        "selected": final_tickers,
        "perf": 0.0 # 다음 회차에서 계산 가능
    })
    with open(STRATEGY_STATE_PATH, 'w') as f:
        json.dump({"trajectory": trajectory[-TRAJECTORY_K:]}, f)

    # Markdown 리포트 생성
    today_str = datetime.now().strftime('%Y-%m-%d')
    filename = f"reports/3S_Trader_Report_{today_str}.md"
    os.makedirs("reports", exist_ok=True)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 3S-Trader KR 전략 리포트 ({today_str})\n\n")
        f.write(f"## 🧠 1. Strategy (Adaptive Strategy)\n{current_strategy}\n\n")
        f.write(f"## 🎯 2. Selection (Top {MAX_PORTFOLIO_STOCKS})\n")
        selected_data = [s for s in scored_universe if s['ticker'] in final_tickers]
        f.write(pd.DataFrame([{
            "종목명": s['name'], "티커": s['ticker'], "현재가": s['data']['price'], "Total점수": sum(s['scores'].values())
        } for s in selected_data]).to_markdown(index=False))
        f.write("\n\n## 📊 3. Scoring Detail (Universe)\n")
        f.write(pd.DataFrame([{
            "종목명": s['name'], "티커": s['ticker'], **s['scores']
        } for s in scored_universe]).to_markdown(index=False))
        f.write("\n\n*본 리포트는 arXiv:2510.17393 논문의 3S 프레임워크를 기반으로 LLM이 직접 분석한 결과입니다.*")

    print(f"리포트 생성 완료: {filename}")

if __name__ == "__main__":
    main()
