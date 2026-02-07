# 3S-Trader KR 📈

이 프로젝트는 논문 **"3S-Trader: A Multi-LLM Framework for Adaptive Stock Scoring, Strategy, and Selection in Portfolio Optimization" (arXiv:2510.17393)**의 핵심 알고리즘을 한국 증시에 맞춰 구현한 AI 투자 보조 시스템입니다.

단순한 기술적 지표 계산을 넘어, **LLM(Large Language Model)**이 시장의 맥락을 이해하고 전문가 수준의 다차원 분석을 수행하는 것이 이 시스템의 핵심입니다.

## 🏗 시스템 아키텍처 및 구현 상세

본 시스템은 논문에서 제시한 3단계 프레임워크(Scoring, Strategy, Selection)를 다음과 같이 충실히 구현하고 있습니다.

### 1. Multi-Dimension Scoring (점수화 모듈)
각 종목의 원천 데이터를 LLM이 직접 읽고 6가지 차원(1~10점)으로 평가합니다.
*   **데이터 소스**: `yfinance`(가격/추세), `네이버 금융`(뉴스 헤드라인)
*   **평가 차원**:
    *   **Financial Health**: 재무 건전성 및 밸류에이션
    *   **Growth Potential**: 시장 점유율 및 성장성
    *   **News Sentiment**: 최근 뉴스에서 나타나는 시장의 긍정/부정적 시각
    *   **News Impact**: 뉴스가 실제 주가에 미칠 영향력의 강도
    *   **Price Momentum**: 단기/중기 가격 추세의 강도
    *   **Volatility Risk**: 변동성에 따른 리스크 수준
*   **구현**: GPT-4o 모델이 뉴스 헤드라인과 기술적 수치를 종합 분석하여 정성적 데이터를 정량적 점수로 변환합니다.

### 2. Adaptive Strategy Generation (전략 모듈)
고정된 가중치를 사용하지 않고, 시장 상황에 따라 최적의 투자 전략을 매번 새롭게 수립합니다.
*   **전략적 피드백 루프**: 시스템은 과거의 선택과 그에 따른 결과(Trajectory)를 기억합니다.
*   **시장 상황 분석**: 현재 시장이 강세장인지, 약세장인지, 혹은 특정 섹터에 쏠림 현상이 있는지를 LLM이 판단합니다.
*   **결과**: "현재는 변동성이 높으니 News Sentiment보다는 Financial Health 비중을 높여 안정적인 종목을 선택하라"와 같은 구체적인 **Selection Strategy**를 생성합니다.

### 3. Final Stock Selection (선택 모듈)
수립된 전략과 종목별 점수를 결합하여 최종 포트폴리오를 구성합니다.
*   **논리적 필터링**: 수립된 전략 텍스트를 LLM이 다시 한번 이해한 후, 30여 개의 후보 종목 중 전략에 가장 부합하는 **TOP 5 종목**을 최종 선발합니다.
*   **현금 비중 조절**: 시장 상황이 극도로 불안정할 경우 현금 비중을 높이도록 전략을 수정할 수 있습니다.

## 📅 리포팅 자동화
*   **매일 오후 7:00 (KST)**: 장 마감 후 최신 데이터를 수집하여 분석을 수행합니다.
*   **GitHub 업로드**: 분석 결과는 `reports/3S_Trader_Report_YYYY-MM-DD.md` 경로에 자동으로 기록됩니다. 리포트에는 **AI가 수립한 전략 전문**과 **종목별 상세 점수**가 포함됩니다.

## 🚀 실행 가이드
1.  **환경 변수 설정**:
    ```bash
    export OPENAI_API_KEY="your_api_key_here"
    ```
2.  **의존성 설치**:
    ```bash
    pip install yfinance pandas requests beautifulsoup4 tabulate
    ```
3.  **수동 실행**:
    ```bash
    python trader.py
    ```

---
*본 프로젝트는 연구 목적으로 제작되었으며, 투자의 최종 결정과 책임은 투자자 본인에게 있습니다.*
