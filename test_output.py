import os
import sys

# Add current directory to path so we can import trader
sys.path.append(os.getcwd())

from trader import _get_stock_data
import json

def test_data_fetch():
    # 삼성전자(KOSPI) 또는 에코프로비엠(KOSDAQ) 중 하나로 테스트
    test_ticker = "247540.KQ" # 에코프로비엠
    print(f"Testing _get_stock_data for {test_ticker}...")
    
    result = _get_stock_data(test_ticker)
    
    # 예쁘게 출력
    print("\n--- [실제 리턴 데이터 구조] ---")
    print(json.dumps(result, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    test_data_fetch()
