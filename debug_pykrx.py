from pykrx import stock
from datetime import datetime, timedelta

def check_pykrx():
    # today = 20260207 (Sat)
    # let's check yesterday (Fri)
    date = "20260205"
    print(f"Checking data for {date}...")
    
    try:
        df = stock.get_market_fundamental_by_ticker(date, market="KOSDAQ")
        print(f"KOSDAQ Fundamental Count: {len(df)}")
        if "247540" in df.index:
            print(f"Ecopro BM Fundamental:\n{df.loc['247540']}")
        else:
            print("Ecopro BM (247540) not found in KOSDAQ fundamentals.")
            # Let's check KOSPI just in case
            df_ks = stock.get_market_fundamental_by_ticker(date, market="KOSPI")
            if "247540" in df_ks.index:
                print("Found in KOSPI!")
    except Exception as e:
        print(f"Error fetching fundamentals: {e}")

    try:
        df_inv = stock.get_market_net_purchases_of_equities_by_ticker("20260130", "20260206", "KOSDAQ")
        print(f"KOSDAQ Investor Count: {len(df_inv)}")
        if "247540" in df_inv.index:
            print(f"Ecopro BM Investor:\n{df_inv.loc['247540']}")
    except Exception as e:
        print(f"Error fetching investors: {e}")

if __name__ == "__main__":
    check_pykrx()
