import os
import sys

# Add current directory to path so we can import trader
sys.path.append(os.getcwd())

from trader import _get_stock_data
import json

def test_data_fetch():
    test_ticker = "247540.KQ" # 에코프로비엠
    print(f"Testing _get_stock_data with News Context for {test_ticker}...")
    
    result = _get_stock_data(test_ticker)
    
    print("\n--- [뉴스 요약 포함 데이터 구조] ---")
    # news_contexts만 따로 확인
    for i, ctx in enumerate(result.get('news_contexts', []), 1):
        print(f"{i}. {ctx[:150]}...")

if __name__ == "__main__":
    test_data_fetch()
