import pandas as pd
from pykrx import stock
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o").strip()
MAX_PORTFOLIO_STOCKS = 30 # User requested Top 30

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        if isinstance(x, (int, float, np.number)): return float(x)
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() in {"nan", "none"}: return None
        return float(s)
    except Exception: return None

def get_latest_trading_day():
    """가장 최근 영업일을 구합니다."""
    today = datetime.now().strftime("%Y%m%d")
    try:
        df = stock.get_market_ohlcv((datetime.now() - timedelta(days=10)).strftime("%Y%m%d"), today, "005930")
        if df.empty: return today
        df = df[df['종가'] > 0]
        return df.index[-1].strftime("%Y%m%d")
    except: return today

def get_stock_universe() -> List[str]:
    """코스닥 시가총액 기준 상위 35개 종목 Universe 구성"""
    target_date = get_latest_trading_day()
    try:
        # Get all tickers' market cap
        df = stock.get_market_cap(target_date)
        if df.empty:
            for i in range(1, 4):
                prev_date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
                df = stock.get_market_cap(prev_date)
                if not df.empty: 
                    target_date = prev_date
                    break
        
        # Filter for KOSDAQ only
        kq_tickers = stock.get_market_ticker_list(target_date, market="KOSDAQ")
        df_kq = df[df.index.isin(kq_tickers)]
        
        top_tickers = df_kq.sort_values(by="시가총액", ascending=False).head(35).index.tolist()
        return top_tickers
    except:
        return ['247540', '191170', '028300', '086520', '291230', '068760', '403870', '058470', '272410', '214150']

def _technical_analysis(ticker: str, target_date: str) -> Optional[Dict[str, Any]]:
    try:
        start_date = (datetime.strptime(target_date, "%Y%m%d") - timedelta(days=100)).strftime("%Y%m%d")
        df = stock.get_market_ohlcv(start_date, target_date, ticker)
        if len(df) < 20: return None
        
        close = df["종가"]
        ma5 = close.rolling(window=5).mean().iloc[-1]
        ma20 = close.rolling(window=20).mean().iloc[-1]
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain.iloc[-1] / loss.iloc[-1]))) if loss.iloc[-1] > 0 else 50
        
        weekly_return = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
        
        return {
            "price": int(close.iloc[-1]),
            "rsi14": float(rsi),
            "weekly_return": float(weekly_return),
            "up_down": "상승" if close.iloc[-1] > close.iloc[-2] else "하락"
        }
    except: return None

def _fetch_news_sentiment(ticker: str) -> int:
    url = f"https://finance.naver.com/item/news_news.naver?code={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        today_str = datetime.now().strftime("%Y.%m.%d")
        news_dates = [td.text.strip() for td in soup.select("td.date")]
        today_news_count = sum(1 for d in news_dates if today_str in d)
        return min(100, 50 + (today_news_count * 5))
    except: return 50

def main():
    print("3S-Trader KR (pykrx + KOSDAQ Focus) Starting...")
    target_date = get_latest_trading_day()
    universe = get_stock_universe()
    
    scored_results = []
    for ticker in universe:
        name = stock.get_market_ticker_name(ticker)
        print(f"Analyzing {name} ({ticker})...")
        
        tech = _technical_analysis(ticker, target_date)
        if not tech: continue
        
        sentiment = _fetch_news_sentiment(ticker)
        fund = stock.get_market_fundamental(target_date, target_date, ticker)
        per = _safe_float(fund["PER"].iloc[-1]) if not fund.empty else 0
        
        # Scoring Logic
        score = (tech['rsi14'] * 0.3) + (tech['weekly_return'] * 0.4) + (sentiment * 0.3)
        
        scored_results.append({
            "Stock Name": name,
            "Ticker": ticker,
            "Price": tech['price'],
            "PER": per,
            "RSI": round(tech['rsi14'], 2),
            "Weekly_Ret%": round(tech['weekly_return'], 2),
            "Sentiment": sentiment,
            "Total_Score": round(score, 2)
        })
        time.sleep(0.05)

    df = pd.DataFrame(scored_results)
    df_sorted = df.sort_values(by="Total_Score", ascending=False)
    portfolio = df_sorted.head(MAX_PORTFOLIO_STOCKS)
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    filename = f"reports/3S_Portfolio_{today_str}.md"
    os.makedirs("reports", exist_ok=True)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 3S-Trader KR KOSDAQ Portfolio Report ({today_str})\n\n")
        f.write(f"- **시장 기준:** KOSDAQ 상위 종목\n- **데이터 기준일:** {target_date}\n\n")
        f.write("## 🎯 AI Selection (Top 30)\n")
        f.write(portfolio.to_markdown(index=False))
        f.write("\n\n*본 리포트는 pykrx 엔진을 사용하여 실시간 코스닥 데이터를 분석한 결과입니다.*")

    print(f"Report generated: {filename}")

if __name__ == "__main__":
    main()
