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
    """Scoring을 위한 고도화된 데이터 수집 (Technical + Fundamental + News + Investor)"""
    code = ticker.split(".")[0]
    market_date = get_latest_trading_day()
    
    # 1. Technical Data (yfinance)
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty: return {}
        
        # 기본 가격 정보
        # pandas Series 비교 에러 방지를 위해 .item() 또는 .iloc[0] 사용
        def get_val(series):
            if isinstance(series, pd.Series):
                return series.iloc[0]
            return series

        last_close = get_val(df['Close'].iloc[-1])
        prev_close = get_val(df['Close'].iloc[-6])
        
        weekly_return = round(((last_close / prev_close) - 1) * 100, 2)
        volatility = round(df['Close'].pct_change().tail(20).std() * 100, 2)
        
        # 이동평균 및 RSI 계산
        ma5 = get_val(df['Close'].rolling(window=5).mean().iloc[-1])
        ma20 = get_val(df['Close'].rolling(window=20).mean().iloc[-1])
        ma60 = get_val(df['Close'].rolling(window=60).mean().iloc[-1])
        
        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        last_rs = get_val(rs.iloc[-1])
        rsi = round(100 - (100 / (1 + last_rs)), 2)
        
        tech_data = {
            "price": int(last_close),
            "weekly_return": weekly_return,
            "volatility": volatility,
            "ma_status": "정배열" if ma5 > ma20 > ma60 else "역배열/혼조",
            "rsi": rsi
        }
    except Exception as e:
        print(f"Technical 수집 오류 ({ticker}): {e}")
        return {}

    # 2. Fundamental Data (pykrx)
    try:
        f_df = stock.get_market_fundamental(market_date, market_date, code)
        fundamental = {
            "per": round(float(f_df['PER'].iloc[-1]), 2) if not f_df.empty else 0,
            "pbr": round(float(f_df['PBR'].iloc[-1]), 2) if not f_df.empty else 0,
            "div_yield": round(float(f_df['배당수익률'].iloc[-1]), 2) if not f_df.empty else 0
        }
    except:
        fundamental = {"per": 0, "pbr": 0, "div_yield": 0}

    # 3. Investor Trends (pykrx - 최근 5거래일 합계)
    try:
        start_date = (datetime.strptime(market_date, "%Y%m%d") - timedelta(days=7)).strftime("%Y%m%d")
        investor_df = stock.get_market_net_purchases_of_equities_by_ticker(start_date, market_date, "KOSDAQ")
        target_inv = investor_df.loc[code] if code in investor_df.index else None
        investor = {
            "foreign": int(target_inv['외국인']) if target_inv is not None else 0,
            "institution": int(target_inv['기관']) if target_inv is not None else 0
        }
    except:
        investor = {"foreign": 0, "institution": 0}

    # 4. News Data (Naver Mobile API)
    news_headlines = []
    url = f"https://m.stock.naver.com/api/news/stock/{code}?pageSize=10&page=1"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            for entry in response.json():
                if 'items' in entry:
                    for item in entry['items']:
                        news_headlines.append(item.get('title', '').replace('&quot;', '"'))
                        if len(news_headlines) >= 5: break
                if len(news_headlines) >= 5: break
    except: pass

    return {**tech_data, **fundamental, **investor, "headlines": news_headlines}

def scoring_agent(ticker: str, data: Dict[str, Any]) -> Dict[str, int]:
    """LLM이 고도화된 데이터를 보고 6개 차원에 대해 점수 산출 (1-10)"""
    if LLM_DISABLED or not data:
        return {d: 5 for d in SCORING_DIMENSIONS}
        
    prompt = f"""Analyze {ticker} with following multidimensional data:
[Technical] Price: {data['price']}, Weekly Return: {data['weekly_return']}%, Volatility: {data['volatility']}%, MA Status: {data['ma_status']}, RSI: {data['rsi']}
[Fundamental] PER: {data['per']}, PBR: {data['pbr']}, Div Yield: {data['div_yield']}%
[Investor] Foreign Net: {data['foreign']}, Institution Net: {data['institution']} (Last 5 days)
[News] Headlines: {data['headlines']}

Assign scores (1-10) for: {SCORING_DIMENSIONS}
Focus on Logical Consistency between Fundamental (Health/Growth) and Technical (Momentum/Risk).

Return ONLY JSON format:
{{"scores": {{"dim_name": score, ...}}, "rationale": "one sentence summary"}}"""

    try:
        res = _openai_chat([{"role": "user", "content": prompt}])
        return _extract_json(res).get("scores", {d: 5 for d in SCORING_DIMENSIONS})
    except:
        return {d: 5 for d in SCORING_DIMENSIONS}

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
