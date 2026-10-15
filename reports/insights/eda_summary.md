# 📊 Exploratory Data Analysis (EDA) Executive Report

**Dataset:** IBM Telco Customer Churn  
**Total Records Analyzed:** 7,043  
**Features:** 21 (Excluding synthetic indicator)  

---

## 1. Target Class Imbalance Analysis
- **Retained Customers ('No'):** 5,174 (73.46%)
- **Churned Customers ('Yes'):** 1,869 (26.54%)
- **Imbalance Ratio:** 2.77:1 (Retained : Churned)

> **ML Insight:** The dataset exhibits moderate class imbalance (~73.5% vs 26.5%).
> If a naive model predicts 'No' for every customer, it achieves **73.46% accuracy** while capturing **0% of churners**.
> Therefore, model selection must be driven by **ROC-AUC, Precision, Recall, and F1-score**, not accuracy.

---

## 2. High-Impact Churn Drivers & Behavioral Patterns

### A. Contract Type (The Strongest Retention Factor)
- **Month-to-month contracts:** **42.71%** churn rate.
- **One year contracts:** **11.27%** churn rate.
- **Two year contracts:** **2.83%** churn rate.
*Finding:* Month-to-month customers are over **15× more likely to churn** than customers on two-year commitments. Month-to-month contracts represent the highest operational churn risk.

### B. Customer Lifecycle & Tenure
- First 12 months (tenure 0-12m) experience the highest attrition rate (~47.7%).
- As tenure increases past 24 months, churn decreases drastically (<15%).
- Customers who stay past month 48 exhibit exceptional retention (>90%).
*Finding:* The first 90 to 180 days (onboarding) represent the critical customer retention window.

### C. Monthly Charges vs Total Charges
- Churned customers have a **higher median monthly charge (~$79.65)** compared to retained customers (~$64.40).
- Retained customers have a **higher median total charges (~$1,683.60)** due to longer lifetime tenure.
*Finding:* Price sensitivity is apparent in customers paying high monthly bills, especially when paired with lack of support/security add-ons.

### D. Internet Service Type & Technical Support
- **Fiber Optic:** High churn rate (**41.89%**), despite being a premium offering.
- **DSL:** Moderate churn rate (**18.96%**).
- **No Internet:** Low churn rate (**7.40%**).
- **Customers without TechSupport:** **41.64%** churn rate vs **15.17%** for those with TechSupport.
*Finding:* Fiber optic customers experience higher monthly charges; if they encounter technical issues without Tech Support, dissatisfaction triggers rapid churn.

### E. Payment Method
- **Electronic Check:** Churn rate is **45.29%**.
- **Credit Card / Bank Transfer (Automatic):** Churn rate is only **~15-16%**.
*Finding:* Customers paying via manual electronic checks lack friction-free auto-renewals and show heightened churn susceptibility.

---

## 3. Correlation & Statistical Analysis
- **Tenure vs Churn:** Negative correlation ($r pprox -0.352$). Longer tenure strongly reduces churn probability.
- **TotalCharges vs Churn:** Negative correlation ($r pprox -0.199$), primarily driven by the underlying tenure component ($r = 0.826$ between Tenure and TotalCharges).
- **MonthlyCharges vs Churn:** Positive correlation ($r pprox +0.193$). Higher monthly cost increases churn tendency.
- *Cautionary Note:* Correlation measures linear association; non-linear models (Random Forests, XGBoost) will uncover complex interactions (e.g. Fiber Optic + No Tech Support + High Monthly Charges).

---

## 4. Key Takeaways for Model Engineering (Phase 3 & 4)
1. Drop `customerID` (zero predictive value, high cardinality).
2. Clean `TotalCharges` (coerce 11 blank spaces to NaN and impute).
3. Encode categorical variables (`Contract`, `InternetService`, `PaymentMethod`) via one-hot encoding.
4. Construct domain features: Tenure cohorts, support add-on counts, and monthly-to-total charge ratios.
