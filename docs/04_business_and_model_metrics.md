# Business & Model Metrics Alignment Framework

**Project Title:** ML Problem Framing & Responsible Data Card  
**Core Objective:** Financial & Technical Metric Optimization for Customer Retention  

---

## 1. Metric Hierarchy & Alignment Matrix

To bridge the gap between technical machine learning evaluation and executive business performance, metrics are structured into a 3-tier hierarchy:

```
                      +--------------------------------------------------+
                      |               TIER 1: BUSINESS METRICS           |
                      |  - Net Saved Recurring Revenue ($)               |
                      |  - Intervention Campaign ROI (%)                 |
                      |  - Customer Lifetime Value (LTV) Retained        |
                      +--------------------------------------------------+
                                               ^
                                               |
                      +--------------------------------------------------+
                      |               TIER 2: FINANCIAL LOSS COST        |
                      |  - Cost-Weighted Loss = FP * C_FP + FN * C_FN    |
                      |  - Cost Ratio: C_FN / C_FP = $500 / $20 = 25     |
                      +--------------------------------------------------+
                                               ^
                                               |
                      +--------------------------------------------------+
                      |               TIER 3: TECHNICAL MODEL METRICS    |
                      |  - Recall (Sensitivity)                          |
                      |  - Precision & F1-Score                          |
                      |  - ROC-AUC & Precision-Recall AUC                |
                      +--------------------------------------------------+
```

---

## 2. Technical Model Metrics vs. Business ROI Metrics

| Metric Level | Metric Name | Mathematical Definition | Target / Benchmark | Operational Business Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Technical** | **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | **$\ge 85.0\%$** | Measures the proportion of actual churners captured by the system. Directly prevents customer loss. |
| **Technical** | **Precision** | $\frac{TP}{TP + FP}$ | $\ge 45.0\%$ | Measures the efficiency of retention offers. Higher precision avoids wasting discounts on non-churners. |
| **Technical** | **F1 Score** | $2 \cdot \frac{\text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$ | $\ge 0.60$ | Harmonic mean balancing recall and precision. |
| **Technical** | **ROC-AUC** | $\int \text{TPR}(f) \, d\text{FPR}(f)$ | $\ge 0.84$ | Overall ranking quality across all classification thresholds independent of class balance. |
| **Financial** | **Financial Loss ($L$)** | $FP \cdot C_{FP} + FN \cdot C_{FN}$ | **Minimize** | Total dollar loss incurred from misclassifications on the evaluation set. |
| **Business** | **Net Retained Revenue** | $TP \cdot (\text{LTV} - C_{FP}) - FP \cdot C_{FP}$ | **Maximize** | Direct net dollar value added by the retention campaign after intervention costs. |
| **Business** | **Intervention ROI (%)** | $\frac{\text{Net Saved Revenue}}{\text{Total Campaign Spend}} \times 100$ | $\ge 300\%$ | Return on investment generated per dollar spent on customer retention offers. |

---

## 3. Explicit Cost Framework: False Positives vs. False Negatives

The core business challenge in churn management stems from extreme asymmetry in error costs.

### A. Cost of a False Positive ($C_{FP}$)
* **Definition:** The model predicts a customer will churn ($\hat{Y} = 1$), but the customer had no intention of leaving ($Y = 0$).
* **Cost Components:**
  1. Promotional retention bill discount: **$15.00**
  2. Administrative & outreach cost (SMS/email/call queue): **$5.00**
* **Total $C_{FP}$ = $20.00 per account**

### B. Cost of a False Negative ($C_{FN}$)
* **Definition:** The model predicts a customer will stay ($\hat{Y} = 0$), but the customer quietly cancels service ($Y = 1$).
* **Cost Components:**
  1. Unmitigated Loss of Customer Lifetime Value (LTV): Average monthly bill $64.76 $\times$ 12 months remaining tenure = **$777.12**
  2. Net replacement credit (offsetting re-acquisition revenue): **-$277.12**
* **Total $C_{FN}$ = $500.00 per lost account**

### Asymmetry Ratio Analysis
$$\text{Cost Ratio } r = \frac{C_{FN}}{C_{FP}} = \frac{\$500}{\$20} = 25$$
A False Negative is **25 times more costly** to the business than a False Positive. Consequently, optimizing a model for raw accuracy ($p = 0.50$) is financially sub-optimal because default thresholds treat $C_{FP}$ and $C_{FN}$ as equal ($1:1$).

---

## 4. Theoretical Threshold Derivation & Empirical Tuning

### Mathematical Optimal Threshold
In decision theory, the optimal probability threshold $p^*$ that minimizes expected financial loss is given by:

$$p^* = \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{20}{20 + 500} = \frac{20}{520} \approx 0.0385 \quad (3.85\%)$$

### Empirical Threshold Tuning Comparison (On 1,409 Test Cases)

To balance operational capacity (call queue limits) against theoretical loss minimization, we evaluate the baseline Logistic Regression model across a spectrum of decision thresholds.

| Decision Threshold ($p$) | Test Accuracy | Precision | Recall | False Positives ($FP$) | False Negatives ($FN$) | Total Financial Loss ($) | Business Assessment |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.50 (Default)** | **80.60%** | **67.12%** | 52.55% | 96 | 177 | $90,420 | High FN rate; misses nearly half of all churners. Financially poor. |
| **0.40** | 79.46% | 60.71% | 63.81% | 154 | 135 | $70,580 | Moderate improvement, but still leaves $67.5k in FN losses. |
| **0.30** | 76.90% | 54.69% | 75.07% | 232 | 93 | $51,140 | Good balance for constrained call queue capacity. |
| **0.20 (Operational)** | **71.22%** | **47.65%** | **86.86%** | **356** | **49** | **$31,620** | **RECOMMENDED:** Captures 86.86% of churners; cuts loss by 65% vs default threshold. |
| **0.10 (Theoretical)**| 58.99% | 38.72% | 93.83% | 554 | 23 | $22,580 | Minimizes loss, but requires large outreach budget for 554 $FP$s. |

### Operational Recommendation & Decision Policy
1. **Primary Operational Threshold ($p^* = 0.20$):** Selected for deployment. Achieves **86.86% Recall** (capturing 324 out of 373 churners in the test set) while keeping False Positives manageable (356 accounts requiring automated $15 bill credit).
2. **Tiered Intervention Policy:**
   * **Tier 1 ($p \ge 0.35$ & Monthly Charges > $70):** Assign to Customer Success Specialist for personal phone outreach.
   * **Tier 2 ($0.20 \le p < 0.35$):** Trigger automated $15 bill credit email campaign.
   * **Tier 3 ($p < 0.20$):** Standard automated lifecycle retention; no financial discount.
