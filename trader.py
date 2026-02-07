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
        
        def get_val(series):
            if hasattr(series, 'iloc'):
                val = series.iloc[-1]
                if hasattr(val, 'iloc'): val = val.iloc[0]
                return float(val)
            return float(series)

        last_close = get_val(df['Close'])
        prev_close = get_val(df['Close'].iloc[-6] if len(df) >= 6 else df['Close'].iloc[0])
        
        weekly_return = round(((last_close / prev_close) - 1) * 100, 2)
        vol_series = df['Close'].pct_change().tail(20).std()
        volatility = round(float(vol_series.iloc[0]) * 100, 2) if hasattr(vol_series, 'iloc') else round(float(vol_series) * 100, 2)
        
        ma5 = get_val(df['Close'].rolling(window=5).mean())
        ma20 = get_val(df['Close'].rolling(window=20).mean())
        ma60 = get_val(df['Close'].rolling(window=60).mean())
        
        # 이격도 (Price vs MA Gaps)
        gap5 = round(((last_close / ma5) - 1) * 100, 2)
        gap20 = round(((last_close / ma20) - 1) * 100, 2)
        
        # 모멘텀 지표 (1개월, 3개월 수익률)
        mom1m = round(((last_close / df['Close'].iloc[-20]) - 1) * 100, 2) if len(df) >= 20 else 0
        mom3m = round(((last_close / df['Close'].iloc[-60]) - 1) * 100, 2) if len(df) >= 60 else 0
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        last_rs = get_val(rs)
        rsi = round(100 - (100 / (1 + last_rs)), 2) if last_rs is not None else 50
        
        tech_data = {
            "price": int(last_close),
            "weekly_return": float(weekly_return),
            "volatility": float(volatility),
            "ma_status": "정배열" if ma5 > ma20 > ma60 else "역배열/혼조",
            "ma_gaps": {"ma5": gap5, "ma20": gap20},
            "momentum": {"1m": mom1m, "3m": mom3m},
            "rsi": float(rsi)
        }
    except Exception as e:
        print(f"Technical 수집 오류 ({ticker}): {e}")
        return {}

    # 2. Fundamental & Investor Data (Naver Scraper - pykrx 대체)
    fundamental = {"per": 0.0, "pbr": 0.0, "div_yield": 0.0}
    investor = {"foreign": 0, "institution": 0}
    
    try:
        # Fundamental (Naver Main)
        url_main = f"https://finance.naver.com/item/main.naver?code={code}"
        res_main = requests.get(url_main, headers={"User-Agent": "Mozilla/5.0"})
        soup_main = BeautifulSoup(res_main.text, "html.parser")
        
        # PER, PBR, 배당수익률 추출 (id 기준)
        def _parse_naver_val(soup, id_str):
            try:
                val = soup.find("em", id=id_str).text.replace(",", "").replace("배", "").replace("%", "")
                return float(val)
            except: return 0.0

        fundamental = {
            "per": _parse_naver_val(soup_main, "_per"),
            "pbr": _parse_naver_val(soup_main, "_pbr"),
            "div_yield": _parse_naver_val(soup_main, "_dvr")
        }

        # Investor (Naver Frgn)
        url_frgn = f"https://finance.naver.com/item/frgn.naver?code={code}"
        res_frgn = requests.get(url_frgn, headers={"User-Agent": "Mozilla/5.0"})
        soup_frgn = BeautifulSoup(res_frgn.text, "html.parser")
        
        # 최근 5거래일 합계 계산
        rows = soup_frgn.select("table.type2 tr")
        f_sum, i_sum = 0, 0
        count = 0
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 9:
                f_buy = cols[6].text.replace(",", "")
                i_buy = cols[5].text.replace(",", "")
                try:
                    f_sum += int(f_buy)
                    i_sum += int(i_buy)
                    count += 1
                except: continue
            if count >= 5: break
            
        investor = {"foreign": f_sum, "institution": i_sum}
    except Exception as e:
        print(f"Naver 데이터 수집 오류 ({ticker}): {e}")

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
                        title = item.get('title', '').replace('&quot;', '"').replace('&amp;', '&')
                        news_headlines.append(title)
                        if len(news_headlines) >= 5: break
                if len(news_headlines) >= 5: break
    except: pass

    return {**tech_data, **fundamental, **investor, "headlines": news_headlines}

def scoring_agent(ticker: str, data: Dict[str, Any]) -> Dict[str, int]:
    """LLM이 고도화된 데이터를 보고 6개 차원에 대해 점수 산출 (1-10)"""
    if LLM_DISABLED or not data:
        return {d: 5 for d in SCORING_DIMENSIONS}
        
    prompt = f"""Analyze {ticker} with following multidimensional data:
[Technical] Price: {data['price']}, Weekly: {data['weekly_return']}%, 1m Mom: {data['momentum']['1m']}%, 3m Mom: {data['momentum']['3m']}%
[MA & RSI] Status: {data['ma_status']}, Gap5: {data['ma_gaps']['ma5']}%, Gap20: {data['ma_gaps']['ma20']}%, RSI: {data['rsi']}
[Fundamental] PER: {data['per']}, PBR: {data['pbr']}, Div Yield: {data['div_yield']}%
[Investor] Foreign Net: {data['foreign']}, Institution Net: {data['institution']} (Last 5 days)
[News] Headlines: {data['headlines']}

Assign scores (1-10) for: {SCORING_DIMENSIONS}
Focus on Technical Momentum (Gap/ROC) and Fundamental Value consistency.

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

def get_market_overview() -> str:
    """시장 전체 상황(코스닥 지수 추세)을 분석하여 텍스트로 리턴합니다."""
    try:
        # 코스닥 지수(101) 데이터 가져오기
        end_date = get_latest_trading_day()
        start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
        
        df = stock.get_market_ohlcv(start_date, end_date, "101", market="KOSDAQ")
        if df.empty: return "Market data unavailable."
        
        curr_idx = df['종가'].iloc[-1]
        prev_idx = df['종가'].iloc[-2]
        change_pct = round(((curr_idx / prev_idx) - 1) * 100, 2)
        
        # 최근 5일 추세
        recent_5 = df['종가'].tail(5)
        trend = "상승" if recent_5.iloc[-1] > recent_5.iloc[0] else "하락/조정"
        
        # 시장 심리 요약
        overview = f"KOSDAQ Index: {curr_idx} ({change_pct}%). "
        overview += f"최근 5거래일 추세는 {trend} 중입니다. "
        
        # 뉴스에서 시장 키워드 추출 (간이)
        url = "https://finance.naver.com/news/mainnews.naver"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        news_titles = [t.text.strip() for t in soup.select(".mainnews_list .articleSubject a")[:3]]
        overview += f" 주요 뉴스: {', '.join(news_titles)}"
        
        return overview
    except Exception as e:
        return f"Market overview error: {e}"

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
    market_overview = get_market_overview()
    current_strategy = strategy_agent(trajectory, market_overview)
    print(f"시장 분석 완료: {market_overview[:50]}...")
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
