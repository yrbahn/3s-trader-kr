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
    '247540.KQ', '191170.KQ', '028300.KQ', '086520.KQ', '291230.KQ',
    '068760.KQ', '403870.KQ', '058470.KQ', '272410.KQ', '214150.KQ',
    '039670.KQ', '145020.KQ', '041190.KQ', '277810.KQ', '066970.KQ',
    '000250.KQ', '121600.KQ', '067310.KQ', '213420.KQ', '091990.KQ',
    '293490.KQ', '035760.KQ', '036540.KQ', '178920.KQ', '196170.KQ',
    '034230.KQ', '051910.KS', '373220.KS', '207940.KS', '005930.KS',
    '000660.KS', '068270.KS', '035420.KS', '035720.KS', '105560.KS'
] # Highly Focused on KOSDAQ (Top 25) + Core KOSPI for comparison

def get_ticker_name(ticker):
    """티커 코드를 종목명으로 변환합니다 (매핑 테이블)"""
    names = {
        # KOSDAQ
        '247540.KQ': '에코프로비엠', '191170.KQ': '알테오젠', '028300.KQ': 'HLB', 
        '086520.KQ': '에코프로', '291230.KQ': '엔켐', '068760.KQ': '셀트리온제약', 
        '403870.KQ': 'HPSP', '058470.KQ': '리노공업', '272410.KQ': '레인보우로보틱스', 
        '214150.KQ': '클래시스', '039670.KQ': '메디톡스', '145020.KQ': '휴젤', 
        '041190.KQ': '우리기술투자', '277810.KQ': '카페24', '066970.KQ': '엘앤에프',
        '000250.KQ': '체시스', '121600.KQ': '나노신소재', '067310.KQ': '양지사',
        '213420.KQ': '덕산네오룩스', '091990.KQ': '셀트리온헬스', '293490.KQ': '카카오게임즈',
        '035760.KQ': 'CJ ENM', '036540.KQ': 'SFA반도체', '178920.KQ': 'SKC솔믹스',
        '196170.KQ': '알서포트', '034230.KQ': '파라다이스',
        # KOSPI
        '051910.KS': 'LG화학', '373220.KS': 'LG엔솔', '207940.KS': '삼성바이오',
        '005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '068270.KS': '셀트리온',
        '035420.KS': 'NAVER', '035720.KS': '카카오', '105560.KS': 'KB금융'
    }
    return names.get(ticker, ticker)

def get_technical_scores(ticker):
    """Scoring Module - Technical Dimension"""
    try:
        data = yf.download(ticker, period="60d", interval="1d", progress=False)
        if data.empty or len(data) < 20: return None
        
        close = data['Close']
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
        if rsi < 30: trend_score += 20 
        if rsi > 70: trend_score -= 20 
        
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
    weights = strategy_module("Neutral")
    results = []
    
    for ticker in STOCK_UNIVERSE:
        name = get_ticker_name(ticker)
        print(f"Scoring {name} ({ticker})...")
        tech = get_technical_scores(ticker)
        if not tech: 
            print(f"Failed to get technical scores for {name}")
            continue
        
        sent = get_sentiment_score(ticker)
        print(f"Sentiment for {name}: {sent}")
        
        # Final Score Calculation
        final_score = (tech['trend_score'] * weights['trend']) + (sent * weights['sentiment'])
        
        results.append({
            'Stock Name': name,
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
    portfolio = df.sort_values(by='Total_Score', ascending=False).head(30)
    cols = ['Stock Name', 'Ticker', 'Price', 'Trend', 'Sentiment', 'Total_Score']
    portfolio = portfolio[cols]
    df = df[cols]
    
    # 3. Reporting
    today_str = datetime.now().strftime('%Y-%m-%d')
    filename = f"reports/3S_Portfolio_{today_str}.md"
    os.makedirs("reports", exist_ok=True)
    
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
