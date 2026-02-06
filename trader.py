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

# --- Configuration ---
STOCK_UNIVERSE = []

UNIVERSE_SOURCE = os.getenv("UNIVERSE_SOURCE", "KOSDAQ_ALL").strip()
UNIVERSE_CACHE_PATH = os.path.join("state", "kosdaq_universe.json")
UNIVERSE_CACHE_TTL_DAYS = int(os.getenv("UNIVERSE_CACHE_TTL_DAYS", "7"))

STATE_DIR = "state"
STRATEGY_STATE_PATH = os.path.join(STATE_DIR, "strategy_state.json")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o").strip()
LLM_DISABLED = os.getenv("LLM_DISABLED", "0").strip() == "1"

MAX_PORTFOLIO_STOCKS = int(os.getenv("MAX_PORTFOLIO_STOCKS", "5"))
TRAJECTORY_K = int(os.getenv("TRAJECTORY_K", "10"))

SCORING_DIMENSIONS = [
    "financial_health",
    "growth_potential",
    "news_sentiment",
    "news_impact",
    "price_momentum",
    "volatility_risk",
]

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

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        return float(s)
    except Exception:
        return None


def _load_json_file(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_json_file(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _is_cache_fresh(path: str, ttl_days: int) -> bool:
    try:
        if not os.path.exists(path):
            return False
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return datetime.now() - mtime < timedelta(days=ttl_days)
    except Exception:
        return False


def _fetch_kosdaq_universe_from_krx() -> List[str]:
    url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&marketType=kosdaqMkt"
    try:
        tables = pd.read_html(url, header=0)
        if not tables:
            return []
        df = tables[0]
        code_col = None
        for c in df.columns:
            if str(c).strip() in {"종목코드", "종목 코드", "Code"}:
                code_col = c
                break
        if code_col is None:
            return []
        codes = (
            df[code_col]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(6)
            .tolist()
        )
        return [f"{c}.KQ" for c in codes if c and c.isdigit()]
    except Exception:
        return []


def get_stock_universe() -> List[str]:
    if UNIVERSE_SOURCE.upper() == "KOSDAQ_ALL":
        if _is_cache_fresh(UNIVERSE_CACHE_PATH, UNIVERSE_CACHE_TTL_DAYS):
            cached = _load_json_file(UNIVERSE_CACHE_PATH)
            if isinstance(cached, dict) and isinstance(cached.get("tickers"), list):
                tickers = [str(x) for x in cached["tickers"] if isinstance(x, str)]
                if tickers:
                    return tickers

        tickers = _fetch_kosdaq_universe_from_krx()
        if tickers:
            _save_json_file(UNIVERSE_CACHE_PATH, {"asof": datetime.now().strftime("%Y-%m-%d"), "tickers": tickers})
            return tickers

        cached = _load_json_file(UNIVERSE_CACHE_PATH)
        if isinstance(cached, dict) and isinstance(cached.get("tickers"), list):
            return [str(x) for x in cached["tickers"] if isinstance(x, str)]
        return []

    return list(STOCK_UNIVERSE)


def _ensure_state_dir() -> None:
    os.makedirs(STATE_DIR, exist_ok=True)


def _load_strategy_state() -> Dict[str, Any]:
    _ensure_state_dir()
    if not os.path.exists(STRATEGY_STATE_PATH):
        return {
            "trajectory": [],
            "current_strategy": "",
            "last_run_date": "",
        }
    try:
        with open(STRATEGY_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"trajectory": [], "current_strategy": "", "last_run_date": ""}
        data.setdefault("trajectory", [])
        data.setdefault("current_strategy", "")
        data.setdefault("last_run_date", "")
        return data
    except Exception:
        return {"trajectory": [], "current_strategy": "", "last_run_date": ""}


def _save_strategy_state(state: Dict[str, Any]) -> None:
    _ensure_state_dir()
    with open(STRATEGY_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _extract_json(text: str) -> Any:
    if not isinstance(text, str):
        raise ValueError("Non-string response")
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)
    m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(1))
    raise ValueError("No JSON found")


def _openai_chat(messages: List[Dict[str, str]], *, temperature: float = 0.2, max_tokens: int = 800) -> str:
    if LLM_DISABLED:
        raise RuntimeError("LLM_DISABLED")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    res = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    res.raise_for_status()
    data = res.json()
    return data["choices"][0]["message"]["content"]


def _llm_json(messages: List[Dict[str, str]], *, temperature: float = 0.2, max_tokens: int = 800, retries: int = 2) -> Any:
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            content = _openai_chat(messages, temperature=temperature, max_tokens=max_tokens)
            return _extract_json(content)
        except Exception as e:
            last_err = e
            time.sleep(0.6)
    raise RuntimeError(f"Failed to get valid JSON from LLM: {last_err}")


def _download_prices(ticker: str, period: str = "6mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data is None or data.empty:
            return None
        return data
    except Exception:
        return None


def _compute_weekly_return(close: pd.Series) -> Optional[float]:
    try:
        close = close.dropna()
        if len(close) < 8:
            return None
        start = close.iloc[-6]
        end = close.iloc[-1]
        if start == 0:
            return None
        return float(end / start - 1.0)
    except Exception:
        return None


def _technical_features(ticker: str) -> Optional[Dict[str, Any]]:
    data = _download_prices(ticker, period="6mo", interval="1d")
    if data is None or len(data) < 40:
        return None
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if len(close) < 40:
        return None

    ma5 = close.rolling(window=5).mean().iloc[-1]
    ma20 = close.rolling(window=20).mean().iloc[-1]
    ma60 = close.rolling(window=60).mean().iloc[-1] if len(close) >= 60 else close.rolling(window=20).mean().iloc[-1]
    current_price = close.iloc[-1]

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs.iloc[-1])) if _safe_float(rs.iloc[-1]) is not None else None

    daily_ret = close.pct_change().dropna()
    vol_20 = float(daily_ret.tail(20).std()) if len(daily_ret) >= 20 else float(daily_ret.std())

    weekly_ret = _compute_weekly_return(close)

    return {
        "price": float(current_price),
        "ma5": float(ma5),
        "ma20": float(ma20),
        "ma60": float(ma60),
        "rsi14": None if rsi is None else float(rsi),
        "volatility_20d": vol_20,
        "weekly_return": weekly_ret,
    }


def _fetch_news_headlines(ticker: str, max_items: int = 8) -> List[str]:
    code = ticker.split(".")[0]
    url = f"https://finance.naver.com/item/news_news.naver?code={code}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        items = []
        for a in soup.select("table.type5 a"):
            title = a.get_text(strip=True)
            if title and title not in items:
                items.append(title)
            if len(items) >= max_items:
                break
        return items
    except Exception:
        return []


def _fundamental_snapshot(ticker: str) -> Dict[str, Any]:
    try:
        t = yf.Ticker(ticker)
        info = getattr(t, "info", {}) or {}
        keys = [
            "marketCap",
            "trailingPE",
            "forwardPE",
            "priceToBook",
            "returnOnEquity",
            "debtToEquity",
            "revenueGrowth",
            "earningsGrowth",
            "profitMargins",
            "grossMargins",
        ]
        out: Dict[str, Any] = {}
        for k in keys:
            if k in info:
                out[k] = info.get(k)
        return out
    except Exception:
        return {}


def _heuristic_scores(overview: Dict[str, Any]) -> Dict[str, Any]:
    tech = overview.get("technical", {}) or {}
    fund = overview.get("fundamental", {}) or {}
    news = overview.get("news", {}) or {}

    weekly_return = _safe_float(tech.get("weekly_return"))
    vol = _safe_float(tech.get("volatility_20d"))
    rsi = _safe_float(tech.get("rsi14"))
    pe = _safe_float(fund.get("trailingPE"))
    pb = _safe_float(fund.get("priceToBook"))
    rev_g = _safe_float(fund.get("revenueGrowth"))
    headlines = news.get("headlines", []) if isinstance(news.get("headlines"), list) else []

    def clamp_1_10(v: float) -> int:
        return int(max(1, min(10, round(v))))

    price_momentum = 5.0
    if weekly_return is not None:
        price_momentum = 5.0 + (weekly_return * 20.0)

    volatility_risk = 5.0
    if vol is not None:
        volatility_risk = 5.0 + (vol * 40.0)

    news_impact = 3.0 + min(7.0, len(headlines) / 2.0)
    news_sentiment = 5.0
    if rsi is not None:
        news_sentiment = 6.0 if rsi < 35 else (4.0 if rsi > 70 else 5.0)

    financial_health = 5.0
    if pe is not None and pe > 0:
        financial_health += 1.0 if pe < 20 else (-1.0 if pe > 40 else 0.0)
    if pb is not None and pb > 0:
        financial_health += 1.0 if pb < 3 else (-1.0 if pb > 10 else 0.0)

    growth_potential = 5.0
    if rev_g is not None:
        growth_potential += max(-2.0, min(3.0, rev_g * 10.0))

    scores = {
        "financial_health": clamp_1_10(financial_health),
        "growth_potential": clamp_1_10(growth_potential),
        "news_sentiment": clamp_1_10(news_sentiment),
        "news_impact": clamp_1_10(news_impact),
        "price_momentum": clamp_1_10(price_momentum),
        "volatility_risk": clamp_1_10(volatility_risk),
    }
    return {
        "scores": scores,
        "rationale": "heuristic",
    }


def _news_agent(ticker: str, headlines: List[str]) -> Dict[str, Any]:
    if LLM_DISABLED:
        return {
            "summary": "",
            "headlines": headlines,
        }

    name = get_ticker_name(ticker)
    joined = "\n".join([f"- {h}" for h in headlines])
    system = "You are a financial news analyst. Return only valid JSON." 
    user = (
        f"Stock: {name} ({ticker})\n"
        f"Recent headlines:\n{joined}\n\n"
        "Task: Summarize recent news sentiment and notable drivers in 3-6 sentences. "
        "Return JSON with keys: summary (string), sentiment (one of: positive, neutral, negative), confidence (0-1)."
    )
    data = _llm_json([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], max_tokens=500)
    if not isinstance(data, dict):
        raise RuntimeError("News agent returned non-dict")
    data["headlines"] = headlines
    return data


def _fundamental_agent(ticker: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    if LLM_DISABLED:
        return {
            "summary": "",
            "snapshot": snapshot,
        }

    name = get_ticker_name(ticker)
    system = "You are a fundamental analyst. Return only valid JSON." 
    user = (
        f"Stock: {name} ({ticker})\n"
        f"Available fundamentals (may be incomplete):\n{json.dumps(snapshot, ensure_ascii=False)}\n\n"
        "Task: Provide a concise fundamental analysis for portfolio selection. "
        "Return JSON with keys: summary (string), strengths (array of strings), risks (array of strings)."
    )
    data = _llm_json([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], max_tokens=650)
    if not isinstance(data, dict):
        raise RuntimeError("Fundamental agent returned non-dict")
    data["snapshot"] = snapshot
    return data


def _technical_agent(ticker: str, features: Dict[str, Any]) -> Dict[str, Any]:
    if LLM_DISABLED:
        return {
            "summary": "",
            "features": features,
        }

    name = get_ticker_name(ticker)
    system = "You are a technical analysis expert. Return only valid JSON." 
    user = (
        f"Stock: {name} ({ticker})\n"
        f"Computed technical features:\n{json.dumps(features, ensure_ascii=False)}\n\n"
        "Task: Summarize trend/momentum and risk in 3-6 sentences, grounded in the features. "
        "Return JSON with keys: summary (string), trend (one of: bullish, neutral, bearish), confidence (0-1)."
    )
    data = _llm_json([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], max_tokens=500)
    if not isinstance(data, dict):
        raise RuntimeError("Technical agent returned non-dict")
    data["features"] = features
    return data


def _score_agent(ticker: str, overview: Dict[str, Any]) -> Dict[str, Any]:
    if LLM_DISABLED:
        return _heuristic_scores(overview)

    name = get_ticker_name(ticker)
    system = "You are the Score Agent in a portfolio framework. Return only valid JSON." 
    user = (
        f"Stock: {name} ({ticker})\n"
        f"Data overview (news/fundamental/technical):\n{json.dumps(overview, ensure_ascii=False)}\n\n"
        "Task: Score the stock from 1 to 10 for each dimension: "
        "Financial Health, Growth Potential, News Sentiment, News Impact, Price Momentum, Volatility Risk. "
        "Higher Volatility Risk means MORE volatile / riskier. "
        "Return JSON with keys: scores (object with keys financial_health, growth_potential, news_sentiment, news_impact, price_momentum, volatility_risk as integers 1-10), rationale (string)."
    )
    data = _llm_json([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], temperature=0.1, max_tokens=700)
    if not isinstance(data, dict) or "scores" not in data:
        raise RuntimeError("Score agent returned invalid")
    scores = data.get("scores")
    if not isinstance(scores, dict):
        raise RuntimeError("Score agent scores invalid")
    for k in SCORING_DIMENSIONS:
        v = scores.get(k)
        if not isinstance(v, int):
            try:
                scores[k] = int(v)
            except Exception:
                scores[k] = 5
        scores[k] = max(1, min(10, int(scores[k])))
    data["scores"] = scores
    return data


def _market_overview(index_ticker: str = "^KS11") -> Dict[str, Any]:
    data = _download_prices(index_ticker, period="6mo", interval="1d")
    if data is None or data.empty:
        return {"index": index_ticker, "weekly_return": None, "volatility_20d": None}
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    weekly_ret = _compute_weekly_return(close)
    daily_ret = close.pct_change().dropna()
    vol_20 = float(daily_ret.tail(20).std()) if len(daily_ret) >= 20 else float(daily_ret.std())
    return {"index": index_ticker, "weekly_return": weekly_ret, "volatility_20d": vol_20}


def _strategy_agent(current_strategy: str, trajectory: List[Dict[str, Any]], market: Dict[str, Any]) -> str:
    if LLM_DISABLED:
        if current_strategy.strip():
            return current_strategy
        return "Prefer financially healthy, lower-volatility stocks; avoid high volatility exposure when index volatility rises."

    traj = trajectory[-TRAJECTORY_K:]
    system = "You are the Strategy Agent for portfolio construction. Return only valid JSON." 
    user = (
        f"Market snapshot: {json.dumps(market, ensure_ascii=False)}\n\n"
        f"Current strategy (may be empty): {current_strategy}\n\n"
        f"Historical trajectory (recent first): {json.dumps(traj, ensure_ascii=False)}\n\n"
        "Task: Propose an updated selection strategy as a short text instruction. "
        "Strategy should refer to the 6 dimensions (financial_health, growth_potential, news_sentiment, news_impact, price_momentum, volatility_risk). "
        "Return JSON with keys: strategy (string), emphasis (object mapping dimension->weight importance 0-1, sum=1)."
    )
    data = _llm_json([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], temperature=0.3, max_tokens=650)
    if not isinstance(data, dict) or "strategy" not in data:
        raise RuntimeError("Strategy agent returned invalid")
    return str(data["strategy"]).strip()


def _selector_agent(strategy: str, scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    if LLM_DISABLED:
        rows = []
        for s in scored:
            scores = s.get("scores", {})
            if not isinstance(scores, dict):
                continue
            base = 0.0
            for k in SCORING_DIMENSIONS:
                v = _safe_float(scores.get(k))
                if v is None:
                    v = 5.0
                if k == "volatility_risk":
                    base += (11.0 - v)
                else:
                    base += v
            rows.append((base, s))
        rows.sort(key=lambda x: x[0], reverse=True)
        picks = rows[:MAX_PORTFOLIO_STOCKS]
        if not picks:
            return {"portfolio": [], "cash_weight": 1.0, "rationale": "no_data"}
        w = 1.0 / len(picks)
        port = [{"ticker": p[1]["ticker"], "weight": round(w, 4)} for p in picks]
        return {"portfolio": port, "cash_weight": 0.0, "rationale": "heuristic"}

    system = "You are the Selector Agent. Return only valid JSON." 
    user = (
        f"Selection strategy: {strategy}\n\n"
        "Candidate scores (each has ticker, name, scores object with 6 dims 1-10, and optional rationale):\n"
        f"{json.dumps(scored, ensure_ascii=False)}\n\n"
        f"Task: Select up to {MAX_PORTFOLIO_STOCKS} stocks and assign weights. Cash holding is allowed. "
        "Constraints: total_stock_weight <= 1, cash_weight = 1 - total_stock_weight, all weights >= 0, at most 5 nonzero stock weights. "
        "Prefer stocks with high scores in dimensions emphasized by the strategy, and avoid high volatility_risk unless strategy explicitly allows. "
        "Return JSON with keys: portfolio (array of {ticker, weight}), cash_weight (number), rationale (string)."
    )
    data = _llm_json([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], temperature=0.2, max_tokens=850)
    if not isinstance(data, dict):
        raise RuntimeError("Selector agent returned invalid")
    port = data.get("portfolio", [])
    if not isinstance(port, list):
        port = []
    cleaned = []
    for item in port:
        if not isinstance(item, dict):
            continue
        t = str(item.get("ticker", "")).strip()
        w = _safe_float(item.get("weight"))
        if not t or w is None:
            continue
        if w < 0:
            continue
        cleaned.append({"ticker": t, "weight": float(w)})
    cleaned = cleaned[:MAX_PORTFOLIO_STOCKS]
    total = sum([x["weight"] for x in cleaned])
    if total > 1.0 and total > 0:
        for x in cleaned:
            x["weight"] = x["weight"] / total
        total = 1.0
    cash_weight = _safe_float(data.get("cash_weight"))
    if cash_weight is None:
        cash_weight = max(0.0, 1.0 - total)
    cash_weight = float(max(0.0, min(1.0, cash_weight)))
    if total + cash_weight > 1.0001:
        cash_weight = max(0.0, 1.0 - total)
    return {"portfolio": cleaned, "cash_weight": cash_weight, "rationale": str(data.get("rationale", "")).strip()}


def _portfolio_return(portfolio: List[Dict[str, Any]], returns_by_ticker: Dict[str, float]) -> float:
    r = 0.0
    for p in portfolio:
        t = p.get("ticker")
        w = _safe_float(p.get("weight"))
        if not t or w is None:
            continue
        rr = returns_by_ticker.get(t)
        if rr is None:
            continue
        r += float(w) * float(rr)
    return float(r)


def _universe_weekly_returns(tickers: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for t in tickers:
        feats = _technical_features(t)
        if not feats:
            continue
        wr = _safe_float(feats.get("weekly_return"))
        if wr is None:
            continue
        out[t] = float(wr)
        time.sleep(0.05)
    return out

def main():
    print("3S-Trader KR Framework Starting...")
    if not LLM_DISABLED and not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required unless LLM_DISABLED=1")

    universe = get_stock_universe()
    if not universe:
        raise RuntimeError("Stock universe is empty. Check KRX connectivity or UNIVERSE_SOURCE.")

    state = _load_strategy_state()
    trajectory = state.get("trajectory", [])
    if not isinstance(trajectory, list):
        trajectory = []
    current_strategy = str(state.get("current_strategy", "")).strip()

    market = _market_overview("^KS11")
    strategy = _strategy_agent(current_strategy, trajectory, market)

    scored_rows: List[Dict[str, Any]] = []
    overview_rows: List[Dict[str, Any]] = []
    prices_cache: Dict[str, Dict[str, Any]] = {}

    for ticker in universe:
        name = get_ticker_name(ticker)
        print(f"Analyzing {name} ({ticker})...")

        tech_feats = _technical_features(ticker)
        if not tech_feats:
            print(f"Failed to get technical data for {name}")
            continue
        prices_cache[ticker] = tech_feats

        headlines = _fetch_news_headlines(ticker, max_items=8)
        fund = _fundamental_snapshot(ticker)

        news_report = _news_agent(ticker, headlines)
        fund_report = _fundamental_agent(ticker, fund)
        tech_report = _technical_agent(ticker, tech_feats)

        overview = {
            "ticker": ticker,
            "name": name,
            "news": news_report,
            "fundamental": fund_report,
            "technical": tech_report,
        }
        overview_rows.append(overview)

        score_res = _score_agent(ticker, overview)
        scores = score_res.get("scores", {})
        if not isinstance(scores, dict):
            print(f"Failed scoring for {name}")
            continue

        row = {
            "ticker": ticker,
            "name": name,
            "price": float(tech_feats.get("price")),
            "scores": scores,
            "rationale": score_res.get("rationale", ""),
        }
        scored_rows.append(row)
        time.sleep(0.1)

    if not scored_rows:
        print("No scoring results found. Check data sources.")
        return

    selection = _selector_agent(strategy, scored_rows)
    portfolio = selection.get("portfolio", [])
    cash_weight = selection.get("cash_weight", 0.0)

    weekly_returns = _universe_weekly_returns([r["ticker"] for r in scored_rows])
    universe_avg = float(np.mean(list(weekly_returns.values()))) if weekly_returns else 0.0
    portfolio_ret = _portfolio_return(portfolio if isinstance(portfolio, list) else [], weekly_returns)

    traj_entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "strategy": strategy,
        "universe_avg_return": universe_avg,
        "portfolio_return": portfolio_ret,
    }
    trajectory.append(traj_entry)
    trajectory = trajectory[-TRAJECTORY_K:]

    state["trajectory"] = trajectory
    state["current_strategy"] = strategy
    state["last_run_date"] = datetime.now().strftime("%Y-%m-%d")
    _save_strategy_state(state)

    report_rows = []
    for r in scored_rows:
        scores = r.get("scores", {})
        report_rows.append({
            "Stock Name": r.get("name"),
            "Ticker": r.get("ticker"),
            "Price": int(float(r.get("price", 0))),
            "FinancialHealth": scores.get("financial_health"),
            "GrowthPotential": scores.get("growth_potential"),
            "NewsSentiment": scores.get("news_sentiment"),
            "NewsImpact": scores.get("news_impact"),
            "PriceMomentum": scores.get("price_momentum"),
            "VolatilityRisk": scores.get("volatility_risk"),
        })

    df = pd.DataFrame(report_rows)
    df = df.sort_values(by=["PriceMomentum", "FinancialHealth"], ascending=False)

    portfolio_table = []
    if isinstance(portfolio, list):
        for p in portfolio:
            t = p.get("ticker")
            w = _safe_float(p.get("weight"))
            if not t or w is None:
                continue
            name = get_ticker_name(t)
            portfolio_table.append({"Stock Name": name, "Ticker": t, "Weight": round(float(w), 4)})
    if cash_weight is None:
        cash_weight = 0.0
    portfolio_table.append({"Stock Name": "CASH", "Ticker": "CASH", "Weight": round(float(cash_weight), 4)})
    portfolio_df = pd.DataFrame(portfolio_table)

    today_str = datetime.now().strftime('%Y-%m-%d')
    filename = f"reports/3S_Portfolio_{today_str}.md"
    os.makedirs("reports", exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 3S-Trader KR Portfolio Report ({today_str})\n\n")
        f.write("> **Paper Ref:** 3S-Trader (arXiv:2510.17393)\n")
        f.write("> **Strategy:** Scoring, Strategy, and Selection for KRX\n\n")
        f.write("## 🎯 Today's Selection (Portfolio)\n")
        f.write(portfolio_df.to_markdown(index=False))
        f.write("\n\n")
        f.write("## 🧠 Strategy\n")
        f.write(strategy.replace("\n", " ") + "\n\n")
        f.write("## 📈 Estimated Weekly Return Snapshot\n")
        f.write(pd.DataFrame([{
            "UniverseAvgWeeklyReturn": universe_avg,
            "PortfolioWeeklyReturn": portfolio_ret,
            "Index": market.get("index"),
            "IndexWeeklyReturn": market.get("weekly_return"),
        }]).to_markdown(index=False))
        f.write("\n\n")
        f.write("## 📊 Scoring Breakdown (Universe)\n")
        f.write(df.to_markdown(index=False))

    print(f"\nReport generated: {filename}")

if __name__ == "__main__":
    main()
