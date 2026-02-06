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
