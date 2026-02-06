# 3S-Trader KR

한국 증시를 위한 **3S-Trader (Scoring, Strategy, Selection)** 프레임워크 구현체입니다.

- Reference paper: **3S-Trader: A Multi-LLM Framework for Adaptive Stock Scoring, Strategy, and Selection in Portfolio Optimization** ([arXiv:2510.17393](https://arxiv.org/abs/2510.17393))

이 저장소는 논문의 핵심 아이디어(점수화/전략/선택 + 전략 반복)를 한국 시장 데이터 소스(yfinance + 네이버 금융) 기반으로 실행 가능한 형태로 구성합니다.

## Framework Architecture

이 구현은 논문의 파이프라인을 다음 단계로 매핑합니다.

### 1) Market Analysis (3개 전문 에이전트)

- **News Agent**
  - 입력: 최근 뉴스 헤드라인
  - 출력: 뉴스 요약/감성(LLM 사용 시 JSON)
- **Fundamental Agent**
  - 입력: `yfinance.Ticker().info` 기반 펀더멘털 스냅샷(가용한 항목만)
  - 출력: 펀더멘털 요약/강점/리스크(LLM 사용 시 JSON)
- **Technical Agent**
  - 입력: 가격 데이터로 계산한 기술 지표(MA, RSI, 변동성, 주간 수익률 등)
  - 출력: 기술적 관점 요약(LLM 사용 시 JSON)

### 2) Stock Scoring (Score Agent)

각 종목에 대해 6개 차원 점수를 **1-10**으로 생성합니다.

- `financial_health`
- `growth_potential`
- `news_sentiment`
- `news_impact`
- `price_momentum`
- `volatility_risk` (점수가 높을수록 더 변동성이 큰/리스크 높은 종목)

LLM을 사용할 수 없는 환경에서는 휴리스틱 기반 백업 스코어링을 수행합니다.

### 3) Strategy (Strategy Agent + Trajectory)

전략은 텍스트 지시문 형태로 유지되며, 실행 시점의 시장 스냅샷 및 과거 성과를 반영해 업데이트됩니다.

- 전략 상태는 `state/strategy_state.json`에 저장됩니다.
- `TRAJECTORY_K`(기본 10)개의 최근 기록을 유지합니다.

### 4) Selection (Selector Agent)

전략과 모든 종목의 점수를 받아 포트폴리오를 구성합니다.

- 최대 종목 수: `MAX_PORTFOLIO_STOCKS` (기본 5)
- 현금 보유 허용 (총 주식 비중 합 ≤ 1)
- 출력: `{ticker, weight}` 리스트 + `cash_weight`

## Installation

```bash
pip install yfinance pandas requests beautifulsoup4 tabulate
```

## Run

### LLM 모드 (논문 구조에 가장 가까움)

OpenAI API Key가 필요합니다.

```bash
export OPENAI_API_KEY="YOUR_KEY"
python3 trader.py
```

### LLM 비활성 모드 (휴리스틱 백업)

```bash
LLM_DISABLED=1 python3 trader.py
```

## Environment Variables

- `OPENAI_API_KEY`: OpenAI API Key (LLM 사용 시 필수)
- `OPENAI_MODEL`: 사용할 모델 (기본값: `gpt-4o`)
- `LLM_DISABLED`: `1`이면 LLM 호출 없이 휴리스틱으로만 실행
- `MAX_PORTFOLIO_STOCKS`: 선택 최대 종목 수 (기본값: `5`)
- `TRAJECTORY_K`: 전략 trajectory 유지 길이 (기본값: `10`)

## Outputs

- `reports/3S_Portfolio_YYYY-MM-DD.md`
  - 포트폴리오(현금 포함)
  - 전략 텍스트
  - 종목별 6차원 점수 테이블
- `state/strategy_state.json`
  - `current_strategy`
  - `trajectory` (최근 K개 기록)

---

이 프로그램은 논문의 아이디어를 기반으로 제작되었으며, 모든 투자의 책임은 본인에게 있습니다.
# 3S-Trader KR 📈

한국 증시를 위한 **3S-Trader (Scoring, Strategy, Selection)** 프레임워크 구현체입니다.  
이 프로젝트는 논문 [arXiv:2510.17393](https://arxiv.org/pdf/2510.17393)의 핵심 방법론을 한국 시장(KOSPI 200 등)에 맞게 최적화하여 구현했습니다.

## 🏗 Framework Architecture

논문에서 제시한 **3S 프레임워크**를 다음과 같이 구현했습니다:

1.  **Scoring Module (점수화 모듈)**:
    *   **Technical Dimension**: `yfinance`를 활용하여 이동평균선(MA), RSI, 모멘텀 등 기술적 지표를 분석하여 0-100점 사이의 점수를 산출합니다.
    *   **Sentiment Dimension**: 네이버 금융 등 주요 뉴스 플랫폼의 뉴스 발행량과 시장 관심을 기반으로 심리 점수를 산출합니다.

2.  **Strategy Module (전략 모듈)**:
    *   시장 상황(강세장, 약세장, 횡보장)을 분석하여 기술적 지표와 심리 지표의 가중치를 동적으로 조절합니다.
    *   예: 강세장에서는 추세(Trend)에 더 높은 가중치를, 불확실한 시장에서는 수급 및 심리(Sentiment)에 가중치를 둡니다.

3.  **Selection Module (선택 모듈)**:
    *   산출된 최종 점수를 바탕으로 전체 Universe(주요 20~50개 종목)에서 최상위 종목을 선정하여 최적의 포트폴리오를 구성합니다.

## 🚀 실행 방법

### 라이브러리 설치
```bash
pip install yfinance pandas requests beautifulsoup4 tabulate
```

### 프로그램 실행
```bash
python trader.py
```

## 📅 리포팅
프로그램 실행 시 `reports/` 폴더에 날짜별 마크다운 리포트가 생성됩니다. 이 리포트에는 당일 선정된 최적 포트폴리오와 각 종목별 점수 데이터가 포함됩니다.

---
*이 프로그램은 논문의 아이디어를 기반으로 제작되었으며, 모든 투자의 책임은 본인에게 있습니다.*
