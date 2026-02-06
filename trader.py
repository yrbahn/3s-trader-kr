import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
from bs4 import BeautifulSoup
import time

# --- Configuration ---
STOCK_UNIVERSE = [
    '005930.KS', '000660.KS', '373220.KS', '005380.KS', '000270.KS',
    '207940.KS', '068270.KS', '035420.KS', '035720.KS', '105560.KS',
    '055550.KS', '005490.KS', '017670.KS', '030200.KS', '259960.KS',
    '352820.KS', '051910.KS', '006400.KS', '012330.KS', '010130.KS'
] # Top 20 for faster execution

def get_technical_scores(ticker):
    """Scoring Module - Technical Dimension"""
    try:
        data = yf.download(ticker, period="60d", interval="1d", progress=False)
        if data.empty or len(data) < 20: return None
        
        close = data['Close']
        # If MultiIndex (happens with some yf versions even for 1 ticker)
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
            
        # Moving Averages
        ma20 = close.rolling(window=20).mean().iloc[-1]
        ma5 = close.rolling(window=5).mean().iloc[-1]
        current_price = close.iloc[-1]
        
        # RSI (14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        
        # Trend Score (0-100)
        trend_score = 50
        if ma5 > ma20: trend_score += 20
        if current_price > ma5: trend_score += 10
        if rsi < 30: trend_score += 20 # Oversold
        if rsi > 70: trend_score -= 20 # Overbought
        
        return {
            'price': int(current_price),
            'rsi': round(rsi, 2),
            'trend_score': min(100, max(0, trend_score))
        }
    except:
        return None

def get_sentiment_score(ticker):
    """Scoring Module - Sentiment Dimension (Simplified)"""
    code = ticker.split('.')[0]
    url = f"https://finance.naver.com/item/main.naver?code={code}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        # Simple heuristic: Look at the 'discussion' count or news headlines
        # For this implementation, we'll just check if the price change is positive as a proxy 
        # (Real implementation would scrape headlines and use LLM)
        news_area = soup.select_one('div.news_section')
        headline_count = len(news_area.select('li')) if news_area else 0
        sentiment_score = 50 + (headline_count * 2)
        return min(100, sentiment_score)
    except:
        return 50

def strategy_module(market_status="Neutral"):
    """Strategy Module - Decides weights based on market condition"""
    if market_status == "Bull":
        return {'trend': 0.7, 'sentiment': 0.3}
    elif market_status == "Bear":
        return {'trend': 0.4, 'sentiment': 0.6}
    else:
        return {'trend': 0.5, 'sentiment': 0.5}

def main():
    print("3S-Trader KR Framework Starting...")
    
    # 1. Strategy Initialization
    # Assume Neutral market for now (could be automated by looking at KOSPI index)
    weights = strategy_module("Neutral")
    
    # 2. Scoring & Selection
    results = []
    for ticker in STOCK_UNIVERSE:
        print(f"Scoring {ticker}...")
        tech = get_technical_scores(ticker)
        if not tech: 
            print(f"Failed to get technical scores for {ticker}")
            continue
        
        sent = get_sentiment_score(ticker)
        print(f"Sentiment for {ticker}: {sent}")
        
        # Final Score Calculation
        final_score = (tech['trend_score'] * weights['trend']) + (sent * weights['sentiment'])
        
        results.append({
            'Ticker': ticker,
            'Price': tech['price'],
            'Trend': tech['trend_score'],
            'Sentiment': sent,
            'Total_Score': final_score
        })
        time.sleep(0.1)

    if not results:
        print("No scoring results found. Check data sources.")
        return

    df = pd.DataFrame(results)
    # Selection: Top 5 Stocks
    portfolio = df.sort_values(by='Total_Score', ascending=False).head(5)
    
    # 3. Reporting
    today_str = datetime.now().strftime('%Y-%m-%d')
    filename = f"reports/3S_Portfolio_{today_str}.md"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 3S-Trader KR Portfolio Report ({today_str})\n\n")
        f.write("> **Paper Ref:** 3S-Trader (arXiv:2510.17393)\n")
        f.write("> **Strategy:** Scoring, Strategy, and Selection for KRX\n\n")
        
        f.write("## 🎯 Today's Top Selection (Portfolio)\n")
        f.write(portfolio.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 📊 Scoring Breakdown (Top 20 Universe)\n")
        f.write(df.sort_values(by='Total_Score', ascending=False).to_markdown(index=False))
        f.write("\n\n*This report uses technical indicators and news volume as scoring signals.*")

    print(f"\nReport generated: {filename}")

if __name__ == "__main__":
    main()
