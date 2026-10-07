# Operational & Algorithmic Risk Register: Telco Churn Prediction System

**Project Title:** ML Problem Framing & Responsible Data Card  
**Scope:** Risk Management Protocol for Deployment & Algorithmic Governance  
**Last Updated:** October 2026  

---

## 1. Risk Matrix Overview

The matrix below maps identified risks across Likelihood and Severity to establish governance priorities for the customer retention machine learning project.

```
       +-------------------------------------------------------------+
       |                        SEVERITY                             |
       |               Low            Medium           High          |
+------+-------------------------------------------------------------+
|  L   | High   |               | R4: Spam     | R5: FN Loss |
|  I   |        |               |   Fatigue    |   ($500 LTV)|
|  K   +--------+---------------+--------------+-------------+
|  E   | Medium | R1: Data      | R2: Fairness | R3: Concept |
|  L   |        |   Leakage     |   Disparity  |   Drift     |
|  I   +--------+---------------+--------------+-------------+
|  H   | Low    | R6: Technical |              |             |
|  O   |        |   Failure     |              |             |
|  O   |        |               |              |             |
|  D   |        |               |              |             |
+------+--------+---------------+--------------+-------------+
```

---

## 2. Comprehensive Risk Register Table

| Risk ID | Risk Title & Description | Affected Stakeholders | Likelihood | Severity | Mitigation Strategy | Assigned Owner & Action Item | Unresolved Questions |
| :---: | :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **R1** | **Data Leakage in Preprocessing:** Computing feature scaling parameters ($\mu, \sigma$) or one-hot encodings across the full dataset prior to train/test splitting. | Data Science Team, Business Leadership | Medium | Low | Codify pipeline transformation logic inside strict leak-free wrappers. Fit scalers strictly on training fold `X_train`. | **Lead Data Scientist:** Implement automated unit tests enforcing zero overlap between train and test distributions. | Are future feature engineering pipelines guaranteed to run inside leak-free pipelines? |
| **R2** | **Demographic Disparity & Bias:** Model predictions exhibit uneven precision/recall across demographic subgroups (`SeniorCitizen`, `gender`). Senior citizens (41.68% raw churn) may receive disproportionate or aggressive digital marketing. | Senior Citizens, Customer Success Team | Medium | Medium | Implement demographic parity audits across subgroups. Establish equal opportunity constraints ensuring high-risk seniors receive human success manager support. | **Responsible AI Lead:** Conduct quarterly fairness audit across `SeniorCitizen` and `gender` slices. | What threshold adjustments are legally permitted under regional telecom non-discrimination laws? |
| **R3** | **Concept & Temporal Drift:** The dataset lacks explicit calendar timestamps. Macroeconomic shifts (e.g., competitor price wars, inflation) cause live churn behavior to diverge from offline training data. | Retention Operations, Finance Team | Medium | High | Implement continuous model monitoring tracking prediction distribution drift (PSI > 0.1) and feature drift (KS-test $p < 0.05$). Retrain model monthly. | **ML Ops Engineer:** Deploy automated model drift monitoring alerts in production pipeline. | What is the actual historical decay rate of subscriber contract stability in live markets? |
| **R4** | **False Positive Discount Cannibalization:** Predicting churn for loyal subscribers ($FP$) triggers unnecessary $15 bill credits, eroding profit margins and training users to expect discounts. | Finance Dept, Marketing Operations | High | Medium | Enforce campaign frequency caps (max 1 offer per customer per 90 days). Restrict direct financial discounts to high-margin accounts; use service add-ons for lower tiers. | **Marketing Ops Lead:** Set hard monthly intervention budget cap ($20,000/month max). | What percentage of $FP$ customers accept discounts without intent to churn? |
| **R5** | **False Negative High-Value Churn:** Failing to predict churn for premium subscribers ($FN$, losing $500–$780 LTV) due to default high classification thresholds ($p = 0.50$). | Executive Leadership, Revenue Teams | High | High | Tune classification decision threshold from default $p = 0.50$ down to cost-optimal threshold ($p^* \approx 0.20$), maximizing recall to 86.86%. | **Lead Data Scientist & Finance:** Standardize cost ratio $C_{FN} / C_{FP} = 25$ in operational policy. | Can customer success specialists handle the increased call queue volume resulting from $p = 0.20$? |
| **R6** | **Silent Technical Pipeline Failure:** Missing features or malformed API payloads cause model scoring service to output fallback predictions (e.g., all 0s), missing active churners. | Customer Success Specialists, End Users | Low | Low | Enforce strict input data schema validation (Pydantic / Great Expectations). Configure automated fallback heuristic rule if scoring service fails. | **Backend Engineer:** Implement automated health-check endpoints and schema validation. | What is the SLA requirement for daily batch scoring completion? |

---

## 3. Deep-Dive Governance Protocols

### Protocol A: Model Drift & Re-calibration Protocol
* **Drift Detection Metric:** Population Stability Index (PSI) calculated weekly on incoming scoring batches against the training baseline:
  $$\text{PSI} = \sum \left( P_{\text{actual}} - P_{\text{baseline}} \right) \times \ln\left( \frac{P_{\text{actual}}}{P_{\text{baseline}}} \right)$$
* **Action Thresholds:**
  * $\text{PSI} < 0.10$: Model stable; no action required.
  * $0.10 \le \text{PSI} \le 0.25$: Warning state; notify ML Ops team and investigate feature drift.
  * $\text{PSI} > 0.25$: Critical drift; automatically trigger retraining pipeline on the most recent 60-day window.

### Protocol B: Algorithmic Fairness & Non-Discrimination Audit
* **Evaluation Metric:** Equal Opportunity Difference (EOD) across sensitive subgroups ($S \in \{\text{SeniorCitizen}=1, \text{SeniorCitizen}=0\}$):
  $$\text{EOD} = |P(\hat{Y}=1 | Y=1, S=1) - P(\hat{Y}=1 | Y=1, S=0)|$$
* **Tolerance Boundary:** If $\text{EOD} > 0.10$, the threshold $p^*$ must be calibrated separately per subgroup to equalize true positive recall across senior and non-senior customer cohorts.
