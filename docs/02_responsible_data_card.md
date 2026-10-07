# Responsible Data Card: Telecom Customer Churn Dataset

**Dataset Title:** Telco Customer Churn Training Dataset  
**File Identifier:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`  
**Dataset Version:** 1.0 (Raw Snapshot)  
**Primary Domain:** Telecommunications Customer Subscription & Churn Analysis  

---

## 1. Data Provenance, Source, & Collection Context

> [!IMPORTANT]
> **Empirical Boundaries & Known Facts**  
> To adhere to strict responsible AI guidelines, facts established directly by the supplied file are explicitly separated from unknown provenance parameters.

### Established Facts (Verified from Source File)
* **File Format & Volume:** Commercially formatted CSV file containing 7,043 rows and 21 columns.
* **Target Label:** Binary outcome column `Churn` with values `"Yes"` (1,869 rows, 26.54%) and `"No"` (5,174 rows, 73.46%).
* **Feature Schema:** 20 predictor attributes capturing customer demographics, contract terms, subscribed service add-ons, and payment billing details.

### Explicit Unknowns & Documented Unresolved Facts
* **Data Origin & Provider:** The filename prefix `WA_Fn-UseC_` matches naming patterns of IBM sample datasets ("Telco Customer Churn"). However, the exact operating telecom entity, parent organization, and geographic region (e.g., US, Canada, Europe) are **unstated**.
* **Temporal Period:** No calendar dates, year identifiers, or snapshot timestamps are present in the dataset. While `tenure` is provided in elapsed months (0 to 72), the calendar era of collection is **unknown**.
* **Sampling & Collection Methodology:** It is **unknown** whether this dataset represents a full census of subscribers, a random sample, or a stratified sub-sample over-representing churned customers.
* **Consent & Privacy Disclosures:** It is **unknown** whether subscribers provided explicit opt-in consent for secondary algorithmic scoring, or whether data collection occurred under standard Terms of Service disclosures.

---

## 2. Dataset Schema & Attribute Categorization

The dataset contains 21 attributes. Sensitive demographic attributes are explicitly highlighted below to ensure fairness monitoring during model development.

| Field Name | Data Type | Value Range / Unique Values | Category | Sensitive Attribute? | Description |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `customerID` | String | 7,043 unique strings (e.g., `7590-VHVEG`) | Identifier | No | Pseudonymized alphanumeric customer account identifier. |
| `gender` | String | `Female` (3,488), `Male` (3,555) | Demographic | **YES** | Self-reported customer gender binary classification. |
| `SeniorCitizen` | Integer | `0` (Non-Senior: 5,901), `1` (Senior: 1,142) | Demographic | **YES** | Binary indicator whether customer is aged 65+. |
| `Partner` | String | `Yes` (3,402), `No` (3,641) | Demographic | **YES** | Marital / domestic partnership status indicator. |
| `Dependents` | String | `Yes` (2,110), `No` (4,933) | Demographic | **YES** | Indicator whether customer has child/family dependents. |
| `tenure` | Integer | 0 to 72 months | Usage | No | Number of elapsed months customer has stayed with company. |
| `PhoneService` | String | `Yes`, `No` | Service | No | Whether customer subscribes to basic phone service. |
| `MultipleLines` | String | `Yes`, `No`, `No phone service` | Service | No | Phone service line extension subscription. |
| `InternetService` | String | `DSL`, `Fiber optic`, `No` | Service | No | Primary internet service infrastructure provider type. |
| `OnlineSecurity` | String | `Yes`, `No`, `No internet service` | Add-On | No | Subscribed cybersecurity / malware protection service. |
| `OnlineBackup` | String | `Yes`, `No`, `No internet service` | Add-On | No | Subscribed cloud data backup service. |
| `DeviceProtection` | String | `Yes`, `No`, `No internet service` | Add-On | No | Subscribed hardware warranty / replacement insurance. |
| `TechSupport` | String | `Yes`, `No`, `No internet service` | Add-On | No | Subscribed premium technical support SLA. |
| `StreamingTV` | String | `Yes`, `No`, `No internet service` | Add-On | No | Subscribed television streaming package. |
| `StreamingMovies` | String | `Yes`, `No`, `No internet service` | Add-On | No | Subscribed movie streaming package. |
| `Contract` | String | `Month-to-month`, `One year`, `Two year` | Financial | No | Customer commitment contract billing terms. |
| `PaperlessBilling` | String | `Yes`, `No` | Billing | No | Electronic billing statement preference. |
| `PaymentMethod` | String | 4 payment channels (e.g., `Electronic check`) | Billing | No | Customer's primary monthly payment method. |
| `MonthlyCharges` | Float | $18.25 to $118.75 | Financial | No | Current monthly recurring subscription billing amount ($). |
| `TotalCharges` | String | Numeric strings + 11 whitespace blanks | Financial | No | Cumulative historical charges billed to account ($). |
| `Churn` | String | `Yes` (1,869), `No` (5,174) | **Target** | No | Binary label indicating voluntary service termination. |

---

## 3. Data Quality, Missingness, & Preprocessing Audits

### Empirical Missingness Audit
Inspection of the raw file reveals that `TotalCharges` is parsed as an `object` data type due to **11 blank whitespace entries (`" "`)**.

```
Missingness Audit Table:
+-------------------+--------------------+------------------------+------------------------------------+
| Column Name       | Count of Blanks    | Corresponding Tenure   | Root Cause & Imputation Policy     |
+-------------------+--------------------+------------------------+------------------------------------+
| TotalCharges      | 11 records (0.16%) | tenure == 0 months     | New accounts signed up before 1st  |
|                   |                    |                        | billing cycle. Impute = 0.0.       |
| All Other Columns | 0 records (0.00%)  | N/A                    | Complete data coverage.            |
+-------------------+--------------------+------------------------+------------------------------------+
```

### Preprocessing Protocol (To Prevent Data Leakage)
1. **TotalCharges Conversion:** Strip whitespace strings and parse to `float64`. Impute missing values with `0.0` (matching `tenure == 0`).
2. **Encoding:** Apply One-Hot Encoding (`pd.get_dummies`) to nominal categorical variables.
3. **Scaling:** Apply Z-score standardization ($\mu, \sigma$) to numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`), **computing mean and standard deviation strictly on the training set**.

---

## 4. Intended vs. Out-of-Scope Uses

### Intended Uses
* **Voluntary Retention Intervention:** Identifying subscribers at high risk of voluntary churn for proactive customer service outreach or loyalty credit offers ($15 bill discount).
* **Service Quality Analysis:** Understanding macro-correlations between internet infrastructure types (e.g., Fiber optic failure rates) and customer dissatisfaction.
* **Customer Success Resource Planning:** Estimating call queue volume required for retention agents based on predicted high-risk account counts.

### Strictly Out-of-Scope & Prohibited Uses
* **Credit Scoring or Financing Denial:** Using churn probability predictions to deny financing, device financing, or credit terms to customers.
* **Automatic Service Degradation or Termination:** Automatically cancelling customer contracts or throttling network bandwidth based on predicted churn score.
* **Price Discrimination:** Systematically inflating subscription prices for non-churning customers who display low elasticity.
* **Employment or Insurance Underwriting:** Utilizing subscriber attributes or demographic indicators for employment screening or insurance pricing.

---

## 5. Limitations & Privacy Considerations

### Dataset Limitations
1. **Static Cross-Sectional Snapshot:** The data reflects a single temporal snapshot without time-series sequence telemetry. It cannot model dynamic behavioral changes (e.g., sudden drop off in weekly app logins).
2. **Absence of Service Outage Logs:** Network outages, ticket escalation history, and speed test failures are missing, despite being primary real-world drivers of telecom churn.
3. **Imbalanced Target Class:** Churn accounts for 26.54% of records. Standard accuracy metrics will favor the majority class unless cost-weighted or threshold-tuned.

### Privacy & Anonymization Audit
* `customerID` values (e.g., `7590-VHVEG`) act as pseudonyms. Direct PII (full names, phone numbers, home addresses, social security numbers) is absent from the dataset.
* **Recommendation for Deployment:** Hash or salt `customerID` strings in analytics logs to enforce zero-trust access control.

---

## 6. Potential Harms & Mitigations

| Potential Risk / Harm | Affected Subgroup | Severity | Concrete Mitigation Strategy |
| :--- | :--- | :---: | :--- |
| **Disproportionate Intervention Disparity** | Senior Citizens (`SeniorCitizen = 1`) | Medium | Senior citizens exhibit a 41.68% churn rate vs. 23.61% for non-seniors. Audit retention offer distribution to ensure senior accounts receive equal access to customer support without aggressive digital-only hurdles. |
| **Spam / Outreach Fatigue** | Customers with $FP$ prediction | Low | Limit proactive retention outreach to a maximum of 1 contact event per 90-day window to avoid harassing satisfied customers. |
| **Electronic Check Payment Disparity** | Low-income / Electronic Check users | Medium | Electronic check users show a 45.29% churn rate. Ensure retention strategies focus on resolving payment friction (e.g., automated payment setup discounts) rather than punitive fees. |
