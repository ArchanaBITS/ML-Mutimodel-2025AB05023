import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Heart Disease Analysis")
st.sidebar.header("Heart Disease Model Selection")
model_options = ["logistic_reg", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
selected_model_name = st.sidebar.selectbox("Choose ML Model", model_options)
st.title("Heart Disease Model Analysis")

##Model_Dir = 'model'

