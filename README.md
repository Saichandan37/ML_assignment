# Breast Cancer Diagnostic - ML Model Comparison

## a. Problem Statement
The goal of this project is to predict whether a tumor is **Malignant** (1) or **Benign** (0) based on diagnostic measurements from a digitized image of a fine needle aspirate (FNA) of a breast mass. This is a binary classification problem.

## b. Dataset Description
- **Source:** Scikit-Learn (UCI Machine Learning Repository)
- **Features:** 30 (Mean Radius, Texture, Perimeter, Area, Smoothness, etc.)
- **Instances:** 569
- **Target Variable:** `target` (0: Benign, 1: Malignant)

---

## c. Models Used: Comparison Table
==================================================
|                     |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:--------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.9737 | 0.9974 |      0.9722 |   0.9859 | 0.979  | 0.9439 |
| Decision Tree       |     0.9474 | 0.944  |      0.9577 |   0.9577 | 0.9577 | 0.888  |
| kNN                 |     0.9474 | 0.9817 |      0.9577 |   0.9577 | 0.9577 | 0.888  |
| Naive Bayes         |     0.9649 | 0.9974 |      0.9589 |   0.9859 | 0.9722 | 0.9253 |
| Random Forest       |     0.9649 | 0.9953 |      0.9589 |   0.9859 | 0.9722 | 0.9253 |
| XGBoost             |     0.9561 | 0.9908 |      0.9583 |   0.9718 | 0.965  | 0.9064 |
==================================================

---

## d. Model Observations
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Excellent performance on this dataset due to high linear separability of features. |
| **Decision Tree** | Faster to train but slightly more volatile than ensemble methods. |
| **kNN** | Performs well when features are scaled, as it relies on Euclidean distance. |
| **Naive Bayes** | High recall, making it useful for medical diagnostics where missing a case is costly. |
| **Random Forest** | Very robust; reduces variance and avoids overfitting through bagging. |
| **XGBoost** | Typically the top performer; handles complex non-linear relationships with boosting. |