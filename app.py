import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Bank marketing Analysis")
st.sidebar.header("Bank Marketing Model Selection")
model_options = ["logistic_reg", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]

##Model Selection
selected_model_name = st.sidebar.selectbox("Choose ML Model", model_options)
st.title("Bank Marketing Model Analysis")

##Upload Test File
st.subheader("Upload Test Data")
uploaded_file = st.file_uploader("Upload your test CSV file", type="csv")

##Model Load
Model_Dir = 'model'

def load_model(model_name):
    path = os.path.join(Model_Dir, f"{model_name}.pkl")
    return joblib.load(path)

##Predict Model
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    model = load_model(selected_model_name)
    y_pred = model.predict(X_test_scaled)

    ##Evaluation Metrics
    st.subheader("Evaluation Metrics")
    metric1, metric2, metric3, metric4, metric5 = st.columns(5)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)  
    mcc = matthews_corrcoef(y_test, y_pred)
    metric1.metric("Accuracy", f"{acc:.2f}")
    metric2.metric("Precision", f"{prec:.2f}")
    metric3.metric("Recall", f"{rec:.2f}")
    metric4.metric("F1 Score", f"{f1:.2f}")
    metric5.metric("MCC", f"{mcc:.2f}")

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("Test CSV is missing")

