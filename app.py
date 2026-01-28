import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Bank marketing Analysis")
st.sidebar.header("Bank Marketing Model Selection")
model_options = ["logistic_reg", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]

## Model Selection
selected_model_name = st.sidebar.selectbox("Choose ML Model", model_options)
st.title("Bank Marketing Model Analysis")

# --- STEP 1: DOWNLOAD ---
st.header("1. Get Test Data")
# Check if the file exists in the directory
if os.path.exists('test.csv'):
    with open("test.csv", "rb") as file:
        st.download_button(
            label="Download test.csv",
            data=file,
            file_name="test_data_for_prediction.csv",
            mime="text/csv"
        )
    st.success("Test template found! Download it above.")
else:
    # training script didn't save test.csv in this folder
    st.error("The file 'test.csv' was not found in the project folder. Run your training script first.")

# --- STEP 2: UPLOAD ---
st.header("2. Upload & Predict")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

## Model Load Helper
Model_Dir = 'model'

def load_model(model_name):
    path = os.path.join(Model_Dir, f"{model_name}.pkl")
    return joblib.load(path)

## Predict Model Logic
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    # Ensure 'target' exists to calculate metrics
    if 'y' in test_df.columns:
        X_test = test_df.drop('y', axis=1)
        y_test = test_df['y']
        
        # NOTE: For better accuracy, you should load the scaler saved from training
        # instead of fitting a new one on test data.
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)

        try:
            model = load_model(selected_model_name)
            y_pred = model.predict(X_test_scaled)

            ## Evaluation Metrics
            st.subheader(f"Evaluation Metrics: {selected_model_name}")
            metric1, metric2, metric3, metric4, metric5 = st.columns(5)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)  
            mcc = matthews_corrcoef(y_test, y_pred)
            
            metric1.metric("Accuracy", f"{acc:.2f}")
            metric2.metric("Precision", f"{prec:.2f}")
            metric3.metric("Recall", f"{rec:.2f}")
            metric4.metric("F1 Score", f"{f1:.2f}")
            metric5.metric("MCC", f"{mcc:.2f}")

            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
        except FileNotFoundError:
            st.error(f"Model file for {selected_model_name} not found in the 'model/' folder.")
    else:
        st.error("The uploaded CSV must contain a 'target' column for evaluation.")
else:
    st.info("Please upload a CSV file to begin the analysis.")
