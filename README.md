# Credit Scoring with LightGBM, WOE & IV Analysis

## 📌 Overview
This project implements a complete **credit risk prediction pipeline** using the Kaggle [*Give Me Some Credit*](https://www.kaggle.com/c/GiveMeSomeCredit) dataset.  
It combines **Weight of Evidence (WOE)** encoding, **Information Value (IV)** analysis, and a **LightGBM** model to achieve high predictive performance while ensuring interpretability through **SHAP values**.  
The pipeline is designed for **credit scoring applications**, providing both **accurate risk predictions** and **actionable business insights**.

---

## 📂 Dataset
- **Source**: Kaggle’s *Give Me Some Credit* dataset  
- **Objective**: Predict whether a person will experience financial distress in the next two years (`SeriousDlqin2yrs` = 1)  
- **Features**: Demographic and financial indicators such as age, monthly income, number of dependents, delinquency history, etc.  
- **Target**: `SeriousDlqin2yrs` (binary classification)

---

## 🔄 Workflow
1. **Data Preprocessing**
   - Missing value imputation using **median strategy**
   - Removing irrelevant or unstable features  
2. **Feature Transformation**
   - Optimal binning with `scorecardpy`
   - WOE encoding for logistic-friendly modeling
   - IV computation for feature importance screening  
3. **Model Training**
   - **LightGBM** classifier with class imbalance handling
   - Early stopping to prevent overfitting  
4. **Evaluation**
   - AUC-ROC on validation data
   - Performance tracking during training  
5. **Interpretability**
   - **SHAP values** for global & local feature impact analysis
   - Summary plots and force plots for visualization  

---

## 📊 Model Evaluation
- **Metric**: AUC-ROC  
- The LightGBM model achieved an AUC-ROC of 0.86 on Kaggle test set, indicating effective feature engineering and well-tuned hyperparameters.  
- Class imbalance was addressed using the `scale_pos_weight` parameter, improving model stability and performance on minority classes.

---

## 🔍 Interpretability & Insights

- **Individual Predictions**:

  **Client 10683**  
  <p align="center">
    <img src="Client #10683.png" width="800" alt="Client 10683 SHAP Force Plot">
  </p>  
  RevolvingUtilizationOfUnsecuredLines_woe and age_woe strongly reduced the predicted risk, while DebtRatio_woe had a minor positive impact. The prediction score was well below the base value, indicating a low probability of default.

  **Client 8652**  
  <p align="center">
    <img src="Client #8652.png" width="800" alt="Client 8652 SHAP Force Plot">
  </p>  
  age_woe, DebtRatio_woe, and RevolvingUtilizationOfUnsecuredLines_woe all contributed to increasing the predicted risk. The prediction score was above the base value, suggesting a higher-than-average default probability.

---

- **Global Feature Importance**:

  **SHAP Summary Plot**  
  <p align="center">
    <img src="SHAP 1.png" width="800" alt="SHAP Summary Plot">
  </p>  
  The SHAP summary plot shows that RevolvingUtilizationOfUnsecuredLines_woe is the most influential feature, with high values increasing default risk.  
  Features such as NumberOfTime30-59DaysPastDueNotWorse_woe, NumberOfTimes90DaysLate_woe, and DebtRatio_woe also significantly increase the risk when high, while higher age_woe generally reduces the likelihood of default.


---

## 📎 References
- **Dataset**: [Kaggle - Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)  
- **LightGBM Documentation**: [https://lightgbm.readthedocs.io/](https://lightgbm.readthedocs.io/)  
- **SHAP Documentation**: [https://shap.readthedocs.io/](https://shap.readthedocs.io/)

