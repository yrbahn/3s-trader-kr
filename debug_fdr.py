import FinanceDataReader as fdr

def check_fdr():
    print("Checking FinanceDataReader StockListing...")
    df = fdr.StockListing('KOSDAQ')
    print(f"Columns: {df.columns.tolist()}")
    if not df.empty:
        print(f"First row:\n{df.iloc[0]}")
        # Search for Ecopro BM
        target = df[df['Symbol'] == '247540'] if 'Symbol' in df.columns else df[df['Code'] == '247540']
        if not target.empty:
            print(f"Ecopro BM:\n{target.iloc[0]}")
        else:
            print("Ecopro BM not found in Listing.")

if __name__ == "__main__":
    check_fdr()
