# 3S-Trader KR 전략 리포트 (2026-02-08)

## 🧠 1. Strategy
Based on the historical performance data (consistently neutral at 0.0%) and the persistent **Stable** market signal, it appears that previous strategies—ranging from broad diversification to quality-growth—have been too defensive or passive to generate alpha. The recurring selection of `095340.KQ` across all portfolios suggests it is a high-quality anchor, but the surrounding selections have failed to contribute to gains.

To break this stagnation, the strategy must shift from "Protection" to "Active Selection."

**Refined Strategy Recommendation: Sentiment-Backed Momentum**

**Rationale:**
With the market remaining **Stable** and past defensive/balanced strategies yielding flat returns, the focus must shift to identifying specific catalysts that drive price action independent of the broader market. We will pivot to a concentrated strategy that prioritizes **News Sentiment** and **Technical Momentum**, using financial health only as a safety filter rather than a primary selection driver.

**Strategy Dimensions:**

1.  **High-Impact News Sentiment (Primary Driver):**
    *   **Focus:** Instead of general monitoring, prioritize stocks with **recent positive news spikes** (e.g., earnings beats, new contracts, or analyst upgrades). In a stable market, these idiosyncratic events are the strongest predictors of short-term price appreciation.
    *   **Action:** Select stocks where sentiment scores are in the top quartile of the universe.

2.  **Technical Relative Strength (Momentum):**
    *   **Focus:** Filter for stocks exhibiting **positive relative strength** compared to the market index over the last 4 weeks. We seek stocks that are already trending upward, indicating institutional accumulation.
    *   **Action:** Avoid "bargain hunting" for laggards. Invest in stocks trading above their 20-day moving average.

3.  **Fundamental "Safety Floor":**
    *   **Focus:** While deprioritizing deep value metrics, ensure candidates have **positive Operating Cash Flow**. This acts as a quality check to ensure the momentum is supported by business reality, not just speculation.
    *   **Action:** Exclude companies with negative cash flow, regardless of sentiment.

4.  **Concentrated Allocation:**
    *   **Focus:** To move the needle on returns, reduce the number of low-conviction holdings.
    *   **Action:** Allocate higher weights (20-30%) to the top 3 stocks that score highest on the intersection of Sentiment and Momentum, rather than equal-weighting a broad list.

**Target Outcome:**
Construct a tighter, more aggressive portfolio that leverages the stable market backdrop to capitalize on specific stock stories and trends, aiming to break the streak of neutral performance.

## 🎯 2. Selection
| 종목명    | 티커        |   비중(%) |    현재가 |   Total |
|:-------|:----------|--------:|-------:|--------:|
| ISC    | 095340.KQ |      40 | 163500 |      43 |
| 비에이치아이 | 083650.KQ |      30 |  75500 |      38 |

## 📊 3. Scoring Detail
| 종목명      | 티커        |   financial_health |   growth_potential |   news_sentiment |   news_impact |   price_momentum |   volatility_risk |
|:---------|:----------|-------------------:|-------------------:|-----------------:|--------------:|-----------------:|------------------:|
| 에코프로     | 086520.KQ |                  2 |                  6 |                5 |             4 |                7 |                 4 |
| 에코프로비엠   | 247540.KQ |                  8 |                  7 |                4 |             5 |                3 |                 3 |
| 알테오젠     | 196170.KQ |                  2 |                  5 |                4 |             5 |                2 |                 3 |
| 레인보우로보틱스 | 277810.KQ |                  2 |                  2 |                5 |             4 |                3 |                 3 |
| 삼천당제약    | 000250.KQ |                  2 |                  2 |                3 |             3 |                8 |                 4 |
| 에이비엘바이오  | 298380.KQ |                  2 |                  7 |                4 |             5 |                2 |                 3 |
| 코오롱티슈진   | 950160.KQ |                  1 |                  1 |                2 |             3 |                3 |                 2 |
| 리노공업     | 058470.KQ |                  3 |                  6 |                5 |             3 |                4 |                 3 |
| HLB      | 028300.KQ |                  3 |                  6 |                6 |             5 |                3 |                 4 |
| 리가켐바이오   | 141080.KQ |                  2 |                  4 |                3 |             4 |                2 |                 3 |
| 케어젠      | 214370.KQ |                  2 |                  7 |                3 |             4 |                4 |                 3 |
| 펩트론      | 087010.KQ |                  2 |                  6 |                3 |             4 |                2 |                 3 |
| 원익IPS    | 240810.KQ |                  3 |                  7 |                2 |             4 |                4 |                 3 |
| 이오테크닉스   | 039030.KQ |                  2 |                  4 |                3 |             5 |                8 |                 3 |
| 클래시스     | 214150.KQ |                  3 |                  6 |                3 |             4 |                3 |                 3 |
| 메지온      | 140410.KQ |                  1 |                  1 |                2 |             5 |                2 |                 2 |
| 로보티즈     | 108490.KQ |                  2 |                  3 |                2 |             4 |                2 |                 2 |
| HPSP     | 403870.KQ |                  2 |                  3 |                3 |             4 |                7 |                 4 |
| 보로노이     | 310210.KQ |                  3 |                  7 |                3 |             4 |                3 |                 4 |
| 파마리서치    | 214450.KQ |                  2 |                  7 |                4 |             7 |                2 |                 2 |
| ISC      | 095340.KQ |                  8 |                  7 |                9 |             7 |                9 |                 3 |
| 현대무벡스    | 319400.KQ |                  2 |                  3 |                4 |             5 |                3 |                 3 |
| 펄어비스     | 263750.KQ |                  2 |                  4 |                3 |             4 |                4 |                 3 |
| 디앤디파마텍   | 347850.KQ |                  2 |                  7 |                7 |             5 |                2 |                 3 |
| 에임드바이오   | 0009K0.KQ |                  3 |                  8 |                8 |             6 |                3 |                 4 |
| 솔브레인     | 357780.KQ |                  8 |                  7 |                8 |             6 |                3 |                 3 |
| 에스티팜     | 237690.KQ |                  3 |                  7 |                7 |             6 |                4 |                 4 |
| 에스피지     | 058610.KQ |                  8 |                  7 |                6 |             6 |                4 |                 3 |
| 휴젤       | 145020.KQ |                  1 |                  6 |                6 |             5 |                2 |                 2 |
| 셀트리온제약   | 068760.KQ |                  3 |                  6 |                8 |             7 |                4 |                 3 |
| 원익홀딩스    | 030530.KQ |                  7 |                  8 |                7 |             6 |                2 |                 3 |
| 동진쎄미켐    | 005290.KQ |                  2 |                  7 |                9 |             8 |                3 |                 3 |
| 실리콘투     | 257720.KQ |                  2 |                  5 |                6 |             5 |                2 |                 3 |
| 올릭스      | 226950.KQ |                  2 |                  7 |                7 |             6 |                2 |                 3 |
| JYP Ent. | 035900.KQ |                  4 |                  8 |                8 |             7 |                3 |                 4 |
| 에스엠      | 041510.KQ |                  3 |                  7 |                8 |             7 |                3 |                 4 |
| 티씨케이     | 064760.KQ |                  3 |                  7 |                8 |             7 |                4 |                 3 |
| 비에이치아이   | 083650.KQ |                  4 |                  7 |                8 |             7 |                9 |                 3 |
| 유진테크     | 084370.KQ |                  7 |                  7 |                1 |             1 |                2 |                 3 |
| 오름테라퓨틱   | 475830.KQ |                  2 |                  7 |                8 |             5 |                3 |                 4 |
| 고영       | 098460.KQ |                  3 |                  8 |                9 |             7 |                5 |                 4 |
| 태성       | 323280.KQ |                  2 |                  7 |                8 |             6 |                8 |                 4 |
| 파두       | 440110.KQ |                  2 |                  7 |                8 |             7 |               10 |                 2 |
| 주성엔지니어링  | 036930.KQ |                  2 |                  3 |                7 |             8 |                9 |                 3 |
| 하나마이크론   | 067310.KQ |                  3 |                  6 |                3 |             2 |                3 |                 4 |
| 삼표시멘트    | 038500.KQ |                  7 |                  7 |                1 |             1 |               10 |                 2 |
| 엘앤씨바이오   | 290650.KQ |                  8 |                  7 |                2 |             1 |                4 |                 4 |
| 쎄트렉아이    | 099320.KQ |                  2 |                  5 |                7 |             5 |                6 |                 4 |
| 하이젠알앤엠   | 160190.KQ |                  2 |                  7 |                8 |             7 |                2 |                 3 |
| 오스코텍     | 039200.KQ |                  3 |                  5 |                2 |             7 |                3 |                 4 |