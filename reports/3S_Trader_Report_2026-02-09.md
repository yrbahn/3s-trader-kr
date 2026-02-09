# 3S-Trader KR 전략 리포트 (2026-02-09)

## 🧠 1. Strategy
Based on the analysis of the recent performance divergence—specifically the success of the Feb 8th strategy (+5.64%) versus the volatility-induced failure of the Feb 9th strategy (-7.18%)—a refined approach is necessary.

**Performance Analysis:**
1.  **The Success (Feb 8):** The "Catalyst-Driven Value" approach worked because it targeted undervalued companies (Low P/B, P/E) with *actual* fundamental improvements (Earnings Revisions).
2.  **The Failure (Feb 9):** The "Earnings-Catalyst Value Reversion" strategy failed due to a specific flaw in its technical entry criteria. By seeking "Oversold (RSI < 45)" assets without a strict enough volatility filter, the portfolio caught a "falling knife" (Stock 196170.KQ, -28.91%). In a "Stable" market, buying distressed assets introduces unnecessary idiosyncratic risk.

**Refined Strategy Recommendation:**

**Strategy Name: Quality-Anchored Value with Volatility Dampening**

**Rationale:**
To capture the upside of the Feb 8th strategy while eliminating the tail risk seen on Feb 9th, we will maintain the **Value + Earnings** core but replace the "Oversold" entry signal with a **"Price Stability"** filter. We are seeking undervalued companies that are steadily climbing, not distressed assets hoping for a bounce.

**Selection Criteria:**
1.  **Fundamental Engine (Earnings & Profitability):** Continue to prioritize stocks with **Positive Earnings Revisions** and strict **Positive Net Income**. This ensures the business is growing, not shrinking.
2.  **Valuation Guardrails:** Maintain focus on **Low P/E and Low P/B** to ensure a margin of safety.
3.  **Risk Control (The Pivot):**
    *   **Exclude High Volatility:** Filter out stocks with extreme recent standard deviation or those that have suffered a >10% drawdown in the last 5 trading days. We are avoiding "falling knives."
    *   **Technical Trend:** Instead of "Oversold," look for **Neutral to Positive Momentum** (Price > 20-day Moving Average). We want assets participating in the "Stable" market trend, not fighting against it.
4.  **Weighting:** Equal Weight (20%) to minimize single-stock impact.

**Objective:** Generate consistent alpha through undervalued growers while strictly filtering out the downside volatility that caused the previous portfolio's drawdown.

## 📈 2. Performance Tracking (과거 추천 성과)
| 추천일        | 추천종목 (수익률)                                                                                    | 평균수익률   |
|:-----------|:----------------------------------------------------------------------------------------------|:--------|
| 2026-02-08 | 420770.KQ (11.28%), 376300.KQ (4.14%), 036620.KQ (2.3%), 095340.KQ (7.52%), 253590.KQ (2.96%) | 5.64%   |

## 🎯 3. Selection (Today's TOP 5)
| 종목명    | 티커        |   비중 |    현재가 |   Total |
|:-------|:----------|-----:|-------:|--------:|
| ISC    | 095340.KQ |   20 | 174800 |      46 |
| 알테오젠   | 196170.KQ |   20 | 543000 |      45 |
| 에스티팜   | 237690.KQ |   20 | 154600 |      45 |
| 셀트리온제약 | 068760.KQ |   20 |  71150 |      40 |
| 에임드바이오 | 0009K0.KQ |   20 |  52300 |      38 |

## 📊 4. Scoring Detail
| 종목명      | 티커        |   financial_health |   growth_potential |   news_sentiment |   news_impact |   price_momentum |   volatility_risk |
|:---------|:----------|-------------------:|-------------------:|-----------------:|--------------:|-----------------:|------------------:|
| 에코프로     | 086520.KQ |                  3 |                  6 |                7 |             5 |                9 |                 3 |
| 알테오젠     | 196170.KQ |                  7 |                  9 |                9 |             8 |                9 |                 3 |
| 에코프로비엠   | 247540.KQ |                  6 |                  7 |                6 |             4 |                9 |                 4 |
| 레인보우로보틱스 | 277810.KQ |                  2 |                  7 |                6 |             3 |                9 |                 3 |
| 삼천당제약    | 000250.KQ |                  2 |                  7 |                8 |             6 |                9 |                 3 |
| 에이비엘바이오  | 298380.KQ |                  2 |                  6 |                4 |             4 |                3 |                 4 |
| 코오롱티슈진   | 950160.KQ |                  1 |                  1 |                5 |             3 |                3 |                 3 |
| 리노공업     | 058470.KQ |                  6 |                  7 |                1 |             1 |                3 |                 4 |
| HLB      | 028300.KQ |                  8 |                  6 |                3 |             7 |                3 |                 4 |
| 리가켐바이오   | 141080.KQ |                  1 |                  6 |                7 |             5 |                2 |                 2 |
| 케어젠      | 214370.KQ |                  3 |                  6 |                4 |             4 |                8 |                 3 |
| 펩트론      | 087010.KQ |                  1 |                  2 |                3 |             4 |                2 |                 3 |
| 원익IPS    | 240810.KQ |                  8 |                  7 |                1 |             1 |                9 |                 3 |
| 이오테크닉스   | 039030.KQ |                  8 |                  7 |                3 |             4 |                9 |                 3 |
| 메지온      | 140410.KQ |                  1 |                  1 |                1 |             1 |                2 |                 2 |
| 클래시스     | 214150.KQ |                  8 |                  7 |                3 |             4 |                3 |                 3 |
| 로보티즈     | 108490.KQ |                  8 |                  7 |                1 |             1 |                2 |                 2 |
| 보로노이     | 310210.KQ |                  1 |                  3 |                3 |             3 |                8 |                 4 |
| HPSP     | 403870.KQ |                  3 |                  4 |                5 |             3 |                8 |                 4 |
| ISC      | 095340.KQ |                  8 |                  9 |                9 |             7 |               10 |                 3 |
| 디앤디파마텍   | 347850.KQ |                  1 |                  7 |                8 |             6 |                2 |                 3 |
| 파마리서치    | 214450.KQ |                  8 |                  7 |                3 |             5 |                2 |                 2 |
| 펄어비스     | 263750.KQ |                  7 |                  8 |                8 |             7 |                3 |                 4 |
| 에임드바이오   | 0009K0.KQ |                  3 |                  8 |                9 |             7 |                7 |                 4 |
| 현대무벡스    | 319400.KQ |                  7 |                  6 |                4 |             5 |                3 |                 3 |
| 솔브레인     | 357780.KQ |                  7 |                  7 |                7 |             5 |                2 |                 3 |
| 에스티팜     | 237690.KQ |                  9 |                  8 |                9 |             7 |                6 |                 6 |
| 셀트리온제약   | 068760.KQ |                  7 |                  7 |                8 |             6 |                7 |                 5 |
| 에스피지     | 058610.KQ |                  8 |                  9 |                8 |             7 |                2 |                 3 |
| 휴젤       | 145020.KQ |                  3 |                  8 |                8 |             7 |                7 |                 5 |