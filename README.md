# Multimodal Classification & Model Comparison
### ML Assignment 2 

**Name :** S.ARCHANA

**ID :** 2025AB05023

## 1. Project Details 
Project is to Implement multiple classification models - Build an interactive Streamlit web application to demonstrate your models - Deploy
the app on Streamlit Community Cloud (FREE) 

## 2. Dataset
**Source:** UCI - Bank Marketing Dataset

**Objects:** 45,211 Instances

**Features:** 17-20 Features

**Target:** y

## 3. Model
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest (Ensemble)
- XGBoost (Ensemble)

## 4. UI
Streamlit
Streamlit Link : https://ml-mutimodel-2025ab05023-wkqmzs6wqcldimq7vpheiv.streamlit.app/

## 5. Evaluation Metrics

| ML Model Name | Accuracy | Precision | Recall | F1 | MCC | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.90 | 0.65 | 0.34 | 0.45 | 0.42 | 0.90 |
| **Decision Tree** | 0.87 | 0.47 | 0.48 | 0.48 | 0.41 | 0.71 |
| **kNN** | 0.89 | 0.60 | 0.32 | 0.42 | 0.39 | 0.82 |
| **Naive Bayes** | 0.85 | 0.39 | 0.51 | 0.45 | 0.36 | 0.81 |
| **Random Forest (Ensemble)** | 0.90 | 0.75 | 0.21 | 0.32 | 0.36 | 0.92 |
| **XGBoost (Ensemble)** | 0.91 | 0.66 | 0.51 | 0.58 | 0.53 | 0.93 |

## 6. Observations About Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | It provides a very stable but "safe" prediction. While accuracy is high, it often misses complex patterns between age and balance, usually resulting in a lower Recall for the actual subscribers. |
| **Decision Tree** | It has a slightly lower Accuracy (0.87), it is much more aggressive than Logistic Regression, nearly doubling the Recall to 0.48. It is better at "finding" potential customers but makes more mistakes (lower Precision) in the process. |
| **kNN** | It maintains a high Accuracy (0.89) and a very respectable AUC (0.82). Like Logistic Regression, it focuses on high-confidence predictions, resulting in a lower Recall (0.32) |
| **Naive Bayes** | It produced the highest Recall (0.51) among the non-ensemble models, effectively catching over half of the subscribers. This comes at the cost of having the lowest Accuracy (0.85) and Precision (0.39). |
| **Random Forest (Ensemble)** | This model is extremely cautious, achieving the highest Precision (0.75) of all models tested. While it is very likely correct when it predicts a "Yes," it only targets "sure thing" customers, as shown by its low Recall (0.21). |
| **XGBoost (Ensemble)** | Dominating nearly every metric, it provides the highest Accuracy (0.91), AUC (0.93), and MCC (0.53). It successfully balances a strong Recall (0.51) with solid Precision (0.66), making it the most effective tool for this campaign |