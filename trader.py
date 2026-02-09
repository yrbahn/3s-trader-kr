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
from concurrent.futures import ThreadPoolExecutor, as_completed
import OpenDartReader

# --- Configuration ---
STATE_DIR = "state"
STRATEGY_STATE_PATH = os.path.join(STATE_DIR, "strategy_state.json")
ANALYSIS_CACHE_PATH = os.path.join(STATE_DIR, "analysis_cache.json")

# LLM Providers: "openai" or "gemini"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_LITE_MODEL = "gpt-4o-mini"
OPENAI_PRO_MODEL = "gpt-4o"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_LITE_MODEL = "gemini-2.0-flash-lite-preview-02-05" 
GEMINI_PRO_MODEL = "gemini-2.0-flash" 

# DART API Key (User provided)
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

# --- DART Collector Class ---
class DartFinancialCollector:
    def __init__(self, api_key):
        self.dart = OpenDartReader(api_key)
    
    def get_summary(self, corp_name):
        """최근 분기 실적 및 부채비율, 유보율 등 심화 지표를 수집합니다."""
        try:
            current_year = datetime.now().year
            reports = [(current_year, '11014'), (current_year, '11012'), (current_year, '11013'), (current_year - 1, '11011')]
            
            # 1. 분기별 실적 추세 (손익계산서)
            summary_list = []
            debt_ratio = "N/A"
            reserve_ratio = "N/A"
            
            for year, code in reports:
                try:
                    # 손익계산서 데이터
                    df_fin = self.dart.finstate(corp_name, year, reprt_code=code)
                    if df_fin is not None and not df_fin.empty:
                        metrics = {"Period": f"{year}.{code}"}
                        for acc in ['매출액', '영업이익', '당기순이익']:
                            row = df_fin[df_fin['account_nm'].str.contains(acc, na=False)]
                            if not row.empty:
                                val = str(row.iloc[0]['thstrm_amount']).replace(',','')
                                metrics[acc] = f"{int(val):,}" if val and val != '-' else "N/A"
                        summary_list.append(metrics)
                    
                    # 2. 재무 건전성 지표 (재무상태표 - 가장 최신 리포트에서 1회만 추출)
                    if debt_ratio == "N/A":
                        df_all = self.dart.finstate_all(corp_name, year, reprt_code=code)
                        if df_all is not None and not df_all.empty:
                            # 부채총계, 자본총계 추출
                            debt = df_all[df_all['account_nm'].str.contains('부채총계', na=False)]
                            equity = df_all[df_all['account_nm'].str.contains('자본총계', na=False)]
                            if not debt.empty and not equity.empty:
                                d_val = float(str(debt.iloc[0]['thstrm_amount']).replace(',',''))
                                e_val = float(str(equity.iloc[0]['thstrm_amount']).replace(',',''))
                                debt_ratio = f"{round((d_val / e_val) * 100, 2)}%"
                except: continue
                if len(summary_list) >= 4: break

            return {
                "quarterly_trend": summary_list,
                "health_metrics": {"debt_ratio": debt_ratio}
            }
        except: return {"error": "DART lookup failed"}

# --- Helper Functions ---

def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"```$", "", text)
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
        res = requests.post(url, json=payload, timeout=60)
        res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        if not OPENAI_API_KEY: return "{}"
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": target_model, "messages": messages, "temperature": temperature}
        res = requests.post(url, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

def get_latest_trading_day():
    today = datetime.now().strftime("%Y%m%d")
    try:
        df = stock.get_market_ohlcv((datetime.now() - timedelta(days=10)).strftime("%Y%m%d"), today, "005930")
        if df.empty: return today
        return df.index[-1].strftime("%Y%m%d")
    except: return today

def get_stock_universe(limit=30) -> List[str]:
    try:
        df_kq = fdr.StockListing('KOSDAQ')
        df_kq = df_kq.sort_values(by='Marcap', ascending=False).head(limit)
        return [f"{c}.KQ" for c in df_kq['Code'].tolist()]
    except: return ["247540.KQ", "086520.KQ"]

# --- Checkpoint & Performance Systems ---

def load_cache(today_str: str) -> Dict[str, Any]:
    if os.path.exists(ANALYSIS_CACHE_PATH):
        try:
            data = json.load(open(ANALYSIS_CACHE_PATH))
            if data.get("date") == today_str: return data.get("results", {})
        except: pass
    return {}

def save_cache(today_str: str, results: Dict[str, Any]):
    os.makedirs(STATE_DIR, exist_ok=True)
    json.dump({"date": today_str, "results": results}, open(ANALYSIS_CACHE_PATH, 'w'), ensure_ascii=False, indent=2)

def _normalize_scores(raw_scores: Dict[str, Any]) -> Dict[str, int]:
    normalized = {d: 5 for d in SCORING_DIMENSIONS}
    mapping = {
        "financial_health": ["financial_health", "financial", "profitability", "valuation"],
        "growth_potential": ["growth_potential", "growth", "potential"],
        "news_sentiment": ["news_sentiment", "sentiment", "market_sentiment"],
        "news_impact": ["news_impact", "impact", "influence"],
        "price_momentum": ["price_momentum", "momentum", "technical"],
        "volatility_risk": ["volatility_risk", "volatility", "risk", "stability"]
    }
    for target, syns in mapping.items():
        for s in syns:
            if s in raw_scores:
                try: 
                    val = raw_scores[s]
                    if isinstance(val, (int, float)): normalized[target] = int(val); break
                except: pass
    return normalized

def calculate_performance(trajectory: List[Dict]) -> List[Dict]:
    if not trajectory: return []
    print("과거 추천주 실시간 성과 추적 중...")
    all_tickers = set()
    for entry in trajectory:
        selected = entry.get("selected", [])
        if isinstance(selected, list):
            for s in selected:
                code = s.get("stock_code") if isinstance(s, dict) else str(s)
                match = re.search(r'(\d{6}\.K[SQ])', str(code).upper())
                if match: all_tickers.add(match.group(1))
    if not all_tickers: return trajectory
    current_prices = {}
    try:
        data = yf.download(" ".join(list(all_tickers)), period="1d", progress=False)
        for t in all_tickers:
            try:
                price_col = data['Close'][t] if len(all_tickers) > 1 else data['Close']
                if not price_col.empty:
                    val = price_col.iloc[-1]
                    current_prices[t] = float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
            except: continue
    except: pass
    for entry in trajectory:
        total_ret = 0.0; total_weight = 0.0
        selected = entry.get("selected", [])
        if not isinstance(selected, list): continue
        for s in selected:
            if not isinstance(s, dict): continue
            raw_code = s.get("stock_code")
            match = re.search(r'(\d{6}\.K[SQ])', str(raw_code).upper())
            code = match.group(1) if match else str(raw_code)
            buy_price = s.get("buy_price"); weight = s.get("weight", 1)
            if code in current_prices and buy_price and buy_price > 0:
                curr_p = current_prices[code]
                ret = ((curr_p / buy_price) - 1) * 100
                s["current_price"] = int(curr_p); s["return"] = round(ret, 2)
                total_ret += ret * weight; total_weight += weight
        if total_weight > 0: entry["perf"] = round(total_ret / total_weight, 2)
    return trajectory

# --- Core Data Fetcher ---

def _get_stock_data(ticker: str) -> Dict[str, Any]:
    code = ticker.split(".")[0]
    name = stock.get_market_ticker_name(code)
    dart_collector = DartFinancialCollector(DART_API_KEY)
    
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty: return {}
        def safe_get(series, idx=-1):
            val = series.iloc[idx]
            return float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
        last_close = safe_get(df['Close'])
        prev_close = safe_get(df['Close'], -6)
        weekly_return = round(((last_close / prev_close) - 1) * 100, 2)
        vol_series = df['Close'].pct_change().tail(20).std()
        volatility = round(float(vol_series.iloc[0] if hasattr(vol_series, 'iloc') else vol_series) * 100, 2)
        ma5 = safe_get(df['Close'].rolling(window=5).mean())
        ma20 = safe_get(df['Close'].rolling(window=20).mean())
        ma60 = safe_get(df['Close'].rolling(window=60).mean())
        tech_summary = f"Price: {int(last_close)}, Weekly: {weekly_return}%, Vol: {volatility}%, MA: {'Bullish' if ma5>ma20>ma60 else 'Neutral'}, Gaps: MA5:{round(((last_close/ma5)-1)*100,2)}%, MA20:{round(((last_close/ma20)-1)*100,2)}%"
        
        # Fundamental (Naver + DART + Consensus)
        url_main = f"https://finance.naver.com/item/main.naver?code={code}"
        res_main = requests.get(url_main, headers={"User-Agent":"Mozilla/5.0"})
        soup_main = BeautifulSoup(res_main.text, "html.parser")
        
        def _parse(s, i):
            try: return float(s.find("em", id=i).text.replace(",","").replace("배","").replace("%",""))
            except: return 0.0
        fund_data = {"per": _parse(soup_main, "_per"), "pbr": _parse(soup_main, "_pbr"), "div_yield": _parse(soup_main, "_dvr")}
        
        # 증권사 컨센서스 (목표주가) 추출
        target_price = "N/A"
        try:
            tp_tag = soup_main.select_one("table.item_info tr td em")
            if tp_tag: target_price = tp_tag.text.replace(",", "")
        except: pass
        
        # DART Detail
        dart_info = dart_collector.get_summary(name)
        
        # Investor
        soup_frgn = BeautifulSoup(requests.get(f"https://finance.naver.com/item/frgn.naver?code={code}", headers={"User-Agent":"Mozilla/5.0"}).text, "html.parser")
        f_sum, i_sum, count = 0, 0, 0
        for row in soup_frgn.select("table.type2 tr"):
            cols = row.find_all("td")
            if len(cols) >= 9:
                try: f_sum += int(cols[6].text.replace(",","")); i_sum += int(cols[5].text.replace(",","")); count += 1
                except: continue
            if count >= 5: break
        
        news_res = requests.get(f"https://m.stock.naver.com/api/news/stock/{code}?pageSize=5&page=1", headers={"User-Agent":"Mozilla/5.0"}).json()
        news_contexts = [f"[{i.get('title','')}] {i.get('body','')}".replace('&quot;','"') for e in news_res if 'items' in e for i in e['items']]
        
        return {
            "tech_text": tech_summary, 
            "fund_text": f"Basic: {fund_data}, TargetPrice: {target_price}, Detailed: {dart_info}, Investor: F:{f_sum}, I:{i_sum}", 
            "news_text": "\n".join(news_contexts), 
            "price": int(last_close)
        }
    except: return {}

# --- Multi-Agent Pipeline (Exact Paper Prompts) ---

def news_agent(ticker, raw_news):
    prompt = f"You are a financial news analysis agent. Task: Summarize recent news for {ticker}.\nContent: {raw_news}\nProvide a concise weekly summary."
    return _llm_chat([{"role": "user", "content": prompt}], model=LITE_MODEL)

def technical_agent(ticker, tech_text):
    prompt = f"You are a stock price analysis agent. Task: Analyze technical indicators for {ticker}.\nData: {tech_text}\nProvide a technical analysis summary."
    return _llm_chat([{"role": "user", "content": prompt}], model=LITE_MODEL)

def fundamental_agent(ticker, fund_text):
    prompt = f"You are a stock fundamentals analysis agent. Task: Analyze financial performance for {ticker}.\nData: {fund_text}\nProvide a summary of fundamental trends."
    return _llm_chat([{"role": "user", "content": prompt}], model=LITE_MODEL)

def score_agent(ticker, news_summ, fund_summ, tech_anal):
    prompt = f"""You are an expert stock evaluation assistant. Tasked with assessing each stock using three input types: News summary, Fundamental analysis, and Recent price behavior.
**stock**: {ticker}
**News Summary**: {news_summ}
**Fundamental Analysis**: {fund_summ}
**Price and Technical Analysis**: {tech_anal}

Evaluate along 6 dimensions (1-10): financial_health, growth_potential, news_sentiment, news_impact, price_momentum, volatility_risk.
Return ONLY JSON with 'scores' and 'justifications'."""
    try:
        res = _extract_json(_llm_chat([{"role": "user", "content": prompt}], model=LITE_MODEL))
        res['scores'] = _normalize_scores(res.get('scores', {}))
        return res
    except: return {"scores": {d: 5 for d in SCORING_DIMENSIONS}}

def strategy_agent(trajectory, market_overview):
    prompt = f"Strategic Advisor. History: {trajectory}\nMarket: {market_overview}\nTask: Define refined data-driven strategy. Return concise professional text."
    return _llm_chat([{"role": "user", "content": prompt}], model=PRO_MODEL, temperature=0.5)

def selection_agent(strategy, candidates):
    reports = "\n".join([f"- {c['name']} ({c['ticker']}): {c['scores']}" for c in candidates])
    prompt = f"Expert stock-picker. Strategy: {strategy}\nCandidates:\n{reports}\nSelect top 5. Return ONLY JSON with 'selected_stocks' (list of {{stock_code, weight}}) and 'reasoning'."
    try: return _extract_json(_llm_chat([{"role": "user", "content": prompt}], model=PRO_MODEL))
    except: return {"selected_stocks": [{"stock_code": c['ticker'], "weight": 20} for c in candidates[:5]]}

def get_market_overview() -> str:
    try:
        end = get_latest_trading_day(); start = (datetime.strptime(end, "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
        df = stock.get_market_ohlcv(start, end, "101", market="KOSDAQ")
        news = [t.text.strip() for t in BeautifulSoup(requests.get("https://finance.naver.com/news/mainnews.naver", headers={"User-Agent":"Mozilla/5.0"}).text, "html.parser").select(".mainnews_list .articleSubject a")[:3]]
        return f"KOSDAQ Index: {df['종가'].iloc[-1]}. News: {', '.join(news)}"
    except: return "Stable market."

def main():
    print("3S-Trader KR: High-Fidelity Multi-Agent Mode (DART Integrated)")
    today_str = datetime.now().strftime('%Y-%m-%d')
    cache = load_cache(today_str)
    market_overview = get_market_overview()
    trajectory = []
    if os.path.exists(STRATEGY_STATE_PATH):
        try: trajectory = json.load(open(STRATEGY_STATE_PATH)).get("trajectory", [])
        except: pass
    current_strategy = strategy_agent(trajectory, market_overview)
    universe = get_stock_universe(limit=30)
    to_do = [t for t in universe if t not in cache]
    
    def process(t):
        raw = _get_stock_data(t)
        if not raw: return None
        n, te, f = news_agent(t, raw['news_text']), technical_agent(t, raw['tech_text']), fundamental_agent(t, raw['fund_text'])
        res = score_agent(t, n, f, te)
        print(f"[Anal: {t}] Scores: {res.get('scores', {})}")
        return t, {"ticker": t, "name": stock.get_market_ticker_name(t.split('.')[0]), "scores": res.get('scores', {}), "data": {"price": raw['price']}}

    if to_do:
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(process, t) for t in to_do]
            for fut in as_completed(futures):
                r = fut.result()
                if r: cache[r[0]] = r[1]; save_cache(today_str, cache)

    scored_universe = [cache[t] for t in universe if t in cache]
    scored_sorted = sorted(scored_universe, key=lambda x: sum(x['scores'].values()), reverse=True)
    sel_res = selection_agent(current_strategy, scored_sorted[:30])
    final_stocks = sel_res.get("selected_stocks", [])
    price_map = {s['ticker']: s['data']['price'] for s in scored_universe}
    for s in final_stocks: s['buy_price'] = price_map.get(str(s.get('stock_code','')), 0)
    today_entry = {"date": today_str, "strategy": current_strategy, "selected": final_stocks, "perf": 0.0}
    found_idx = next((i for i, e in enumerate(trajectory) if e.get("date") == today_str), -1)
    if found_idx >= 0: trajectory[found_idx] = today_entry
    else: trajectory.append(today_entry)
    trajectory = calculate_performance(trajectory)
    json.dump({"trajectory": trajectory[-TRAJECTORY_K:]}, open(STRATEGY_STATE_PATH, 'w'), ensure_ascii=False, indent=2)

    filename = f"reports/3S_Trader_Report_{today_str}.md"; os.makedirs("reports", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 3S-Trader KR 전략 리포트 ({today_str})\n\n## 🧠 1. Strategy\n{current_strategy}\n\n")
        f.write("## 📈 2. Performance Tracking\n")
        past = [e for e in trajectory if e['date'] != today_str]
        if past:
            p_list = []
            for e in reversed(past):
                picks = []
                for s in e.get('selected', []):
                    c = s.get('stock_code', 'N/A')
                    match = re.search(r'(\d{6}\.K[SQ])', str(c))
                    nm = stock.get_market_ticker_name(match.group(1).split('.')[0]) if match else c
                    picks.append(f"{nm} ({s.get('return', 0)}%)")
                p_list.append({"추천일": e['date'], "추천종목 (수익률)": ", ".join(picks[:5]), "평균수익률": f"{e.get('perf', 0)}%"})
            f.write(pd.DataFrame(p_list).head(10).to_markdown(index=False) + "\n\n")
        else: f.write("*과거 기록 없음*\n\n")
        f.write(f"## 🎯 3. Selection\n")
        selected_data = []
        for s in final_stocks:
            code = str(s.get('stock_code',''))
            match = next((x for x in scored_universe if code in x['ticker'] or x['ticker'] in code), None)
            if match: selected_data.append((match, s.get('weight', 0)))
        if selected_data:
            df = pd.DataFrame([{"종목명": m['name'], "티커": m['ticker'], "비중": w, "현재가": m['data']['price'], "Total": sum(m['scores'].values())} for m, w in selected_data])
            f.write(df.sort_values("비중", ascending=False).to_markdown(index=False) + "\n\n")
        f.write("## 📊 4. Scoring Detail\n")
        f.write(pd.DataFrame([{"종목명": s['name'], "티커": s['ticker'], **s['scores']} for s in scored_universe]).to_markdown(index=False))
    print(f"Report: {filename}")

if __name__ == "__main__": main()
