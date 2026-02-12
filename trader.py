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
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import FinanceDataReader as fdr
from pykrx import stock
from concurrent.futures import ThreadPoolExecutor, as_completed
import OpenDartReader

# --- Configuration ---
STATE_DIR = "state"
STRATEGY_STATE_PATH = os.path.join(STATE_DIR, "strategy_state.json")
DB_PATH = os.path.join(STATE_DIR, "trader_database.db")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_LITE_MODEL = "gpt-4o-mini"
OPENAI_PRO_MODEL = "gpt-4o"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_LITE_MODEL = "gemini-2.5-flash-lite"
GEMINI_PRO_MODEL = "gemini-3-pro-preview"

DART_API_KEY = "6dd44b6c2f494848116618fbc0ea3947196f3ef0"

if LLM_PROVIDER == "gemini":
    LITE_MODEL = GEMINI_LITE_MODEL
    PRO_MODEL = GEMINI_PRO_MODEL
else:
    LITE_MODEL = OPENAI_LITE_MODEL
    PRO_MODEL = OPENAI_PRO_MODEL

LLM_DISABLED = os.getenv("LLM_DISABLED", "0").strip() == "1"
MAX_PORTFOLIO_STOCKS = 5
TRAJECTORY_K = 30 

SCORING_DIMENSIONS = ["financial_health", "growth_potential", "news_sentiment", "news_impact", "price_momentum", "volatility_risk"]

# --- Helper Functions (Defined Early to avoid NameError) ---

def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text); text = re.sub(r"```$", "", text)
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match: return json.loads(match.group(1))
    raise ValueError("No JSON found")

def _llm_chat(messages: List[Dict[str, str]], model: str = None, temperature=0.2) -> str:
    if os.getenv("LLM_DISABLED", "0").strip() == "1": return "{}"
    target_model = model if model else LITE_MODEL
    if LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY: return "{}"
        prompt = "\n".join([m['content'] for m in messages])
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={GEMINI_API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": temperature}}
        res = requests.post(url, json=payload, timeout=60); res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        if not OPENAI_API_KEY: return "{}"
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": target_model, "messages": messages, "temperature": temperature}
        res = requests.post(url, headers=headers, json=payload, timeout=60); res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

# --- 1. RDB Data Collection Pipeline (SQLite) ---

class StockDatabase:
    """SQLite 기반의 원천 데이터베이스 관리 클래스"""
    def __init__(self, db_path):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_data (
                    ticker TEXT PRIMARY KEY,
                    name TEXT,
                    timestamp TEXT,
                    technical TEXT,
                    fundamental TEXT,
                    dart TEXT,
                    investor TEXT,
                    news TEXT
                )
            """)
            conn.commit()

    def update_stock(self, data: Dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO raw_data 
                (ticker, name, timestamp, technical, fundamental, dart, investor, news)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['ticker'], data['name'], datetime.now().isoformat(),
                json.dumps(data['technical'], ensure_ascii=False),
                json.dumps(data['fundamental'], ensure_ascii=False),
                json.dumps(data['dart'], ensure_ascii=False),
                json.dumps(data['investor'], ensure_ascii=False),
                json.dumps(data['news'], ensure_ascii=False)
            ))
            conn.commit()

    def get_stock(self, ticker: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM raw_data WHERE ticker = ?", (ticker,))
            row = cur.fetchone()
            if row:
                res = dict(row)
                for key in ['technical', 'fundamental', 'dart', 'investor', 'news']:
                    res[key] = json.loads(res[key])
                return res
        return None

class DartFinancialCollector:
    def __init__(self, api_key):
        self.dart = OpenDartReader(api_key)
    def get_summary(self, corp_name):
        try:
            current_year = datetime.now().year
            reports = [(current_year, '11014'), (current_year, '11012'), (current_year, '11013'), (current_year - 1, '11011')]
            summary_list, debt_ratio = [], "N/A"
            for year, code in reports:
                try:
                    df_fin = self.dart.finstate(corp_name, year, reprt_code=code)
                    if df_fin is not None and not df_fin.empty:
                        m = {"Period": f"{year}.{code}"}
                        for acc in ['매출액', '영업이익', '당기순이익']:
                            row = df_fin[df_fin['account_nm'].str.contains(acc, na=False)]
                            if not row.empty:
                                val = str(row.iloc[0]['thstrm_amount']).replace(',','')
                                m[acc] = f"{int(val):,}" if val and val != '-' else "N/A"
                        summary_list.append(m)
                    if debt_ratio == "N/A":
                        df_all = self.dart.finstate_all(corp_name, year, reprt_code=code)
                        if df_all is not None and not df_all.empty:
                            debt = df_all[df_all['account_nm'].str.contains('부채총계', na=False)]
                            equity = df_all[df_all['account_nm'].str.contains('자본총계', na=False)]
                            if not debt.empty and not equity.empty:
                                debt_ratio = f"{round((float(str(debt.iloc[0]['thstrm_amount']).replace(',','')) / float(str(equity.iloc[0]['thstrm_amount']).replace(',',''))) * 100, 2)}%"
                except: continue
                if len(summary_list) >= 4: break
            return {"quarterly_trend": summary_list, "debt_ratio": debt_ratio}
        except: return {"error": "DART lookup failed"}

def collect_stock_data(ticker: str) -> Dict[str, Any]:
    code = ticker.split(".")[0]; name = stock.get_market_ticker_name(code)
    dart_collector = DartFinancialCollector(DART_API_KEY)
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty: return {}
        def safe_get(series, idx=-1):
            val = series.iloc[idx]; return float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
        
        last_close = safe_get(df['Close'])
        exp1 = df['Close'].ewm(span=12, adjust=False).mean(); exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2; signal = macd.ewm(span=9, adjust=False).mean()
        macd_status = "Golden Cross" if safe_get(macd) > safe_get(signal) and macd.iloc[-2] <= signal.iloc[-2] else "Bearish" if safe_get(macd) < safe_get(signal) else "Neutral"
        vol_spike = round(float(df['Volume'].iloc[-1] / df['Volume'].rolling(window=20).mean().iloc[-1]), 2)
        
        technical = {"price": int(last_close), "weekly_return": round(((last_close / safe_get(df['Close'], -6)) - 1) * 100, 2), "macd": macd_status, "vol_spike": vol_spike, "is_bullish": safe_get(df['Close'].rolling(window=5).mean()) > safe_get(df['Close'].rolling(window=20).mean())}
        
        soup_main = BeautifulSoup(requests.get(f"https://finance.naver.com/item/main.naver?code={code}", headers={"User-Agent":"Mozilla/5.0"}).text, "html.parser")
        def _p(s, i):
            try: return float(s.find("em", id=i).text.replace(",","").replace("배","").replace("%",""))
            except: return 0.0
        
        fundamental = {"per": _p(soup_main, "_per"), "pbr": _p(soup_main, "_pbr"), "roe": _p(soup_main, "_roe"), "target_price": soup_main.select_one("table.item_info tr td em").text.replace(",", "") if soup_main.select_one("table.item_info tr td em") else "N/A"}
        
        dart = dart_collector.get_summary(name)
        soup_frgn = BeautifulSoup(requests.get(f"https://finance.naver.com/item/frgn.naver?code={code}", headers={"User-Agent":"Mozilla/5.0"}).text, "html.parser")
        f_sum, i_sum = 0, 0
        for row in soup_frgn.select("table.type2 tr")[:15]:
            cols = row.find_all("td")
            if len(cols) >= 9:
                try: f_sum += int(cols[6].text.replace(",","")); i_sum += int(cols[5].text.replace(",",""))
                except: continue
        
        news_res = requests.get(f"https://m.stock.naver.com/api/news/stock/{code}?pageSize=5&page=1", headers={"User-Agent":"Mozilla/5.0"}).json()
        news = [f"[{i.get('title','')}] {i.get('body','')}".replace('&quot;','"') for e in news_res if 'items' in e for i in e['items']]
        
        return {"ticker": ticker, "name": name, "technical": technical, "fundamental": fundamental, "dart": dart, "investor": {"foreign_net": f_sum, "institution_net": i_sum}, "news": news}
    except: return {}

# --- 2. Multi-Agent Logic (Reading from SQLite) ---

def news_agent(raw: Dict) -> str:
    p = f"You are a financial news analysis agent. Task: Summarize news for {raw['name']}.\nContent: {' '.join(raw['news'])}\nProvide a concise summary."
    return _llm_chat([{"role": "user", "content": p}], model=LITE_MODEL)

def technical_agent(raw: Dict) -> str:
    p = f"You are a stock price analysis agent. Task: Analyze technicals for {raw['name']}.\nData: {raw['technical']}\nProvide analysis summary."
    return _llm_chat([{"role": "user", "content": p}], model=LITE_MODEL)

def fundamental_agent(raw: Dict) -> str:
    p = f"You are a stock fundamentals analysis agent. Task: Analyze financial performance for {raw['name']}.\nData: {raw['fundamental']}, DART: {raw['dart']}, Investor: {raw['investor']}\nProvide summary."
    return _llm_chat([{"role": "user", "content": p}], model=LITE_MODEL)

def score_agent(raw: Dict, n_summ: str, f_summ: str, t_anal: str) -> Dict[str, Any]:
    p = f"Expert evaluator. Stock: {raw['name']}\nNews: {n_summ}\nFund: {f_summ}\nTech: {t_anal}\nScore 6 dimensions (1-10): financial_health, growth_potential, news_sentiment, news_impact, price_momentum, volatility_risk.\nReturn ONLY JSON."
    try:
        res = _extract_json(_llm_chat([{"role": "user", "content": p}], model=LITE_MODEL))
        res['scores'] = _normalize_scores(res.get('scores', {}))
        return res
    except: return {"scores": {d: 5 for d in SCORING_DIMENSIONS}}

# --- Common Logic ---

def _normalize_scores(raw_s: Dict) -> Dict[str, int]:
    norm = {d: 5 for d in SCORING_DIMENSIONS}
    m = {"financial_health": ["financial_health", "financial", "profitability", "valuation"], "growth_potential": ["growth_potential", "growth", "potential"], "news_sentiment": ["news_sentiment", "sentiment", "market_sentiment"], "news_impact": ["news_impact", "impact", "influence"], "price_momentum": ["price_momentum", "momentum", "technical"], "volatility_risk": ["volatility_risk", "volatility", "risk", "stability"]}
    for k, syns in m.items():
        for s in syns:
            if s in raw_s:
                try: norm[k] = int(raw_s[s]); break
                except: pass
    return norm

def strategy_agent(traj, overview):
    p = f"Strategic Advisor. History: {traj}\nMarket: {overview}\nTask: Define strategy. Return concise professional text."
    return _llm_chat([{"role": "user", "content": p}], model=PRO_MODEL, temperature=0.5)

def selection_agent(strat, cand):
    reports = "\n".join([f"- {c['name']} ({c['ticker']}): {c['scores']}" for c in cand])
    p = f"Expert stock-picker. Strategy: {strat}\nCandidates:\n{reports}\nSelect top 5. Return ONLY JSON with 'selected_stocks' and 'reasoning'."
    try: return _extract_json(_llm_chat([{"role": "user", "content": p}], model=PRO_MODEL))
    except: return {"selected_stocks": [{"stock_code": c['ticker'], "weight": 20} for c in cand[:5]]}

def get_latest_trading_day():
    today = datetime.now().strftime("%Y%m%d")
    try:
        df = stock.get_market_ohlcv((datetime.now() - timedelta(days=10)).strftime("%Y%m%d"), today, "005930")
        return df.index[-1].strftime("%Y%m%d")
    except: return today

def get_market_overview() -> str:
    try:
        end = get_latest_trading_day(); start = (datetime.strptime(end, "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
        df = stock.get_market_ohlcv(start, end, "101", market="KOSDAQ")
        news = [t.text.strip() for t in BeautifulSoup(requests.get("https://finance.naver.com/news/mainnews.naver", headers={"User-Agent":"Mozilla/5.0"}).text, "html.parser").select(".mainnews_list .articleSubject a")[:3]]
        return f"KOSDAQ: {df['종가'].iloc[-1]}. News: {', '.join(news)}"
    except: return "Stable market."

def calculate_performance(trajectory: List[Dict]) -> List[Dict]:
    if not trajectory: return []
    all_t = set()
    for e in trajectory:
        for s in e.get("selected", []):
            code = s.get("stock_code") if isinstance(s, dict) else str(s)
            match = re.search(r'(\d{6}\.K[SQ])', str(code).upper())
            if match: all_t.add(match.group(1))
    if not all_t: return trajectory
    curr_p = {}
    try:
        data = yf.download(" ".join(list(all_t)), period="1d", progress=False)
        for t in all_t:
            try:
                p_col = data['Close'][t] if len(all_t) > 1 else data['Close']
                if not p_col.empty: curr_p[t] = float(p_col.iloc[-1].iloc[0]) if hasattr(p_col.iloc[-1], 'iloc') else float(p_col.iloc[-1])
            except: continue
    except: pass
    for e in trajectory:
        total_ret, total_w = 0.0, 0.0
        for s in e.get("selected", []):
            if not isinstance(s, dict): continue
            m = re.search(r'(\d{6}\.K[SQ])', str(s.get("stock_code")).upper())
            code = m.group(1) if m else s.get("stock_code")
            buy_p, w = s.get("buy_price"), s.get("weight", 1)
            if code in curr_p and buy_p and buy_p > 0:
                ret = ((curr_p[code] / buy_p) - 1) * 100
                s["current_price"] = int(curr_p[code]); s["return"] = round(ret, 2)
                total_ret += ret * w; total_w += w
        if total_w > 0: e["perf"] = round(total_ret / total_w, 2)
    return trajectory

# --- Main Execution ---

def main():
    print("3S-Trader KR: RDB(SQLite) & Multi-Agent Pipeline Centric Mode")
    today_str = datetime.now().strftime('%Y-%m-%d')
    db = StockDatabase(DB_PATH)
    
    # 1. Collection Phase
    print("Step 1: Pipeline - Collecting raw data to SQLite...")
    universe = [f"{c}.KQ" for c in fdr.StockListing('KOSDAQ').sort_values(by='Marcap', ascending=False).head(30)['Code'].tolist()]
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(collect_stock_data, t): t for t in universe}
        for fut in as_completed(futures):
            raw = fut.result()
            if raw: db.update_stock(raw); print(f" Saved DB: {raw['ticker']}")
    
    # 2. Analysis Phase
    print("\nStep 2: Experts - Analyzing from SQLite...")
    scored_universe = []
    def analyze_task(t):
        raw = db.get_stock(t)
        if not raw: return None
        n, te, f = news_agent(raw), technical_agent(raw), fundamental_agent(raw)
        res = score_agent(raw, n, f, te)
        print(f" Score {t}: {res['scores']}")
        return {"ticker": t, "name": raw['name'], "scores": res['scores'], "price": raw['technical']['price']}

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(analyze_task, t) for t in universe]
        for fut in as_completed(futures):
            r = fut.result()
            if r: scored_universe.append(r)
            
    # 3. Strategy & Selection
    trajectory = []
    if os.path.exists(STRATEGY_STATE_PATH):
        try: trajectory = json.load(open(STRATEGY_STATE_PATH)).get("trajectory", [])
        except: pass
    
    current_strategy = strategy_agent(trajectory, get_market_overview())
    scored_sorted = sorted(scored_universe, key=lambda x: sum(x['scores'].values()), reverse=True)
    sel_res = selection_agent(current_strategy, scored_sorted[:30])
    final_stocks = sel_res.get("selected_stocks", [])
    
    for s in final_stocks:
        raw = db.get_stock(s.get('stock_code',''))
        if raw: s['buy_price'] = raw['technical']['price']
    
    today_entry = {"date": today_str, "strategy": current_strategy, "selected": final_stocks, "perf": 0.0}
    found_idx = next((i for i, e in enumerate(trajectory) if e.get("date") == today_str), -1)
    if found_idx >= 0: trajectory[found_idx] = today_entry
    else: trajectory.append(today_entry)
    
    trajectory = calculate_performance(trajectory)
    json.dump({"trajectory": trajectory[-TRAJECTORY_K:]}, open(STRATEGY_STATE_PATH, 'w'), ensure_ascii=False, indent=2)

    # 4. Report
    filename = f"reports/3S_Trader_Report_{today_str}.md"; os.makedirs("reports", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 3S-Trader KR 전략 리포트 ({today_str})\n\n## 🧠 1. Strategy\n{current_strategy}\n\n## 🎯 3. Selection\n")
        sel_tickers = [s.get('stock_code','') for s in final_stocks]
        final_data = [s for s in scored_universe if s['ticker'] in sel_tickers]
        if final_data: f.write(pd.DataFrame(final_data).to_markdown(index=False) + "\n\n")
        f.write("## 📊 4. Scoring Detail\n")
        f.write(pd.DataFrame(scored_universe).to_markdown(index=False))

    print(f"Report: {filename}")

if __name__ == "__main__": main()
