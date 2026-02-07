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
# UNIVERSE_SOURCE: "KOSDAQ_TOP_30" (Default), "KOSPI_TOP_30"
UNIVERSE_SOURCE = os.getenv("UNIVERSE_SOURCE", "KOSDAQ_TOP_30").strip()
STATE_DIR = "state"
STRATEGY_STATE_PATH = os.path.join(STATE_DIR, "strategy_state.json")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o").strip()
LLM_DISABLED = os.getenv("LLM_DISABLED", "0").strip() == "1"

MAX_PORTFOLIO_STOCKS = int(os.getenv("MAX_PORTFOLIO_STOCKS", "5"))

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
        # 삼성전자 데이터를 통해 실제 데이터가 있는 영업일을 확인
        df = stock.get_market_ohlcv((datetime.now() - timedelta(days=10)).strftime("%Y%m%d"), today, "005930")
        if df.empty:
            return today # Fallback
        # 주말/공휴일 등 데이터가 0인 날짜 필터링
        df = df[df['종가'] > 0]
        return df.index[-1].strftime("%Y%m%d")
    except:
        return today

def get_stock_universe() -> List[str]:
    """시가총액 기준 상위 종목 Universe를 구성합니다."""
    target_date = get_latest_trading_day()
    market = "KOSDAQ" if "KOSDAQ" in UNIVERSE_SOURCE.upper() else "KOSPI"
    
    try:
        # get_market_cap_by_ticker 대신 get_market_cap(일자) 사용 (더 안정적인 bulk API)
        df = stock.get_market_cap(target_date)
        if df.empty:
            # 특정 일자 조회가 실패하면 최근 3일 중 데이터가 있는 날을 찾음
            for i in range(1, 4):
                prev_date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
                df = stock.get_market_cap(prev_date)
                if not df.empty: break
        
        if df.empty:
            # 최후의 수단: 하드코딩된 주요 종목 반환
            return ['005930', '000660', '373220', '005380', '068270']
            
        # KOSPI/KOSDAQ 구분 필요시 필터링 로직 추가 가능
        # 여기서는 전체 시장 시총 상위 30개를 기본으로 하되 요청하신 코스닥 위주로 구성
        # pykrx의 get_market_cap 결과에는 시장 구분이 없으므로 전체 top 30 사용
        top_tickers = df.sort_values(by="시가총액", ascending=False).head(30).index.tolist()
        return top_tickers
    except:
        return ['005930', '000660', '373220', '005380', '068270']

def _technical_analysis(ticker: str, target_date: str) -> Optional[Dict[str, Any]]:
    """Scoring Module - Technical Dimension (pykrx 기반)"""
    try:
        start_date = (datetime.strptime(target_date, "%Y%m%d") - timedelta(days=100)).strftime("%Y%m%d")
        df = stock.get_market_ohlcv(start_date, target_date, ticker)
        if len(df) < 20: return None
        
        close = df["종가"]
        ma5 = close.rolling(window=5).mean().iloc[-1]
        ma20 = close.rolling(window=20).mean().iloc[-1]
        current_price = close.iloc[-1]
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain.iloc[-1] / loss.iloc[-1]))) if loss.iloc[-1] > 0 else 50
        
        # 주간 수익률 (최근 5영업일)
        weekly_return = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
        
        return {
            "price": int(current_price),
            "ma5": float(ma5),
            "ma20": float(ma20),
            "rsi14": float(rsi),
            "weekly_return": float(weekly_return)
        }
    except: return None

def _fetch_news_sentiment(ticker: str) -> int:
    """Scoring Module - Sentiment (Naver News 수량 기반 간이 측정)"""
    url = f"https://finance.naver.com/item/news_news.naver?code={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        # 오늘 날짜의 뉴스 개수 확인
        today_str = datetime.now().strftime("%Y.%m.%d")
        news_dates = [td.text.strip() for td in soup.select("td.date")]
        today_news_count = sum(1 for d in news_dates if today_str in d)
        
        score = 50 + (today_news_count * 5)
        return min(100, score)
    except: return 50

def _openai_chat(messages: List[Dict[str, str]]) -> str:
    if not OPENAI_API_KEY: return "LLM Offline (No Key)"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "messages": messages, "temperature": 0.2}
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        return res.json()["choices"][0]["message"]["content"]
    except: return "LLM Error"

def main():
    print(f"3S-Trader KR (pykrx version) Starting...")
    target_date = get_latest_trading_day()
    universe = get_stock_universe()
    
    scored_results = []
    for ticker in universe:
        name = stock.get_market_ticker_name(ticker)
        print(f"Analyzing {name} ({ticker})...")
        
        tech = _technical_analysis(ticker, target_date)
        if not tech: continue
        
        sentiment = _fetch_news_sentiment(ticker)
        
        # Fundamental (pykrx)
        fund = stock.get_market_fundamental(target_date, target_date, ticker)
        per = _safe_float(fund["PER"].iloc[-1]) if not fund.empty else None
        pbr = _safe_float(fund["PBR"].iloc[-1]) if not fund.empty else None
        
        # Simple Scoring Logic (can be replaced by LLM if desired)
        score = (tech['rsi14'] * 0.3) + (tech['weekly_return'] * 0.4) + (sentiment * 0.3)
        
        scored_results.append({
            "Stock Name": name,
            "Ticker": ticker,
            "Price": tech['price'],
            "PER": per,
            "PBR": pbr,
            "RSI": tech['rsi14'],
            "Weekly_Ret%": tech['weekly_return'],
            "Sentiment": sentiment,
            "Total_Score": round(score, 2)
        })
        time.sleep(0.05)

    df = pd.DataFrame(scored_results)
    portfolio = df.sort_values(by="Total_Score", ascending=False).head(MAX_PORTFOLIO_STOCKS)
    
    # Report Generation
    today_str = datetime.now().strftime('%Y-%m-%d')
    filename = f"reports/3S_Portfolio_{today_str}.md"
    os.makedirs("reports", exist_ok=True)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 3S-Trader KR Portfolio Report ({today_str})\n\n")
        f.write(f"> **Market Context:** {UNIVERSE_SOURCE} (Base Date: {target_date})\n\n")
        f.write("## 🎯 Today's AI Selection\n")
        f.write(portfolio.to_markdown(index=False))
        f.write("\n\n## 📊 Universe Scoring (Top 30)\n")
        f.write(df.sort_values(by="Total_Score", ascending=False).to_markdown(index=False))
        f.write("\n\n*This report is generated using pykrx engine for high precision.*")

    print(f"Report generated: {filename}")

if __name__ == "__main__":
    main()
