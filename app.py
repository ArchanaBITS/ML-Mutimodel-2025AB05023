import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, classification_report

st.set_page_config(page_title="Bank Marketing Analysis", layout="wide")
st.sidebar.header("Model Selection")
model_options = ["logistic_reg", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]

selected_model_name = st.sidebar.selectbox("Choose ML Model", model_options)
st.title("🏦 Bank Marketing Model Analysis")

# DOWNLOAD TEST
st.header("1. Get Test Data")
test_path = 'data/test.csv'
if os.path.exists(test_path):
    with open(test_path, "rb") as file:
        st.download_button(
            label="Download test.csv",
            data=file,
            file_name="test_data_for_prediction.csv",
            mime="text/csv"
        )
    st.success("Test template found!")
else:
    st.error("The file 'data/test.csv' was not found. Please run your training script first.")

# UPLOAD & PREDICT
st.header("2. Upload & Predict")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    if 'y' in test_df.columns:
        y_test = test_df['y']
        X_test = test_df.drop('y', axis=1)

        # --- PREPROCESSING ---
        # One-Hot Encoding
        X_test_encoded = pd.get_dummies(X_test)

        # Match Columns with Training Data
        try:
            model_cols = joblib.load('model/model_columns.pkl')
            X_test_final = X_test_encoded.reindex(columns=model_cols, fill_value=0)
            
            # 3. Load and apply SAVED Scaler
            scaler = joblib.load('model/scaler.pkl')
            X_test_scaled = scaler.transform(X_test_final)

            # --- PREDICTION ---
            model_path = os.path.join('model', f"{selected_model_name}.pkl")
            model = joblib.load(model_path)
            y_pred = model.predict(X_test_scaled)
            
            # AUC requires the probability of the positive class (1)
            if hasattr(model, "predict_proba"):
               y_probs = model.predict_proba(X_test_scaled)[:, 1]
               auc_score = roc_auc_score(y_test, y_probs)
            else:
            # Some models like LinearSVC don't have predict_proba by default
               auc_score = 0.0
            # --- EVALUATION ---
            st.subheader(f"📊 Results for {selected_model_name}")
            model1, model2, model3, model4, model5, model6 = st.columns(6)
            
            model1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
            model2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.2f}")
            model3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.2f}")
            model4.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.2f}")
            model5.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.2f}")
            model6.metric("AUC", f"{roc_auc_score(y_test, y_probs):.2f}")
            
            if hasattr(model, "feature_importances_"):
               st.subheader("🌲 Feature Importance")
    
               # Create a DataFrame for visualization
               feat_importances = pd.Series(model.feature_importances_, index=X_test_final.columns)
               top_feats = feat_importances.nlargest(10)
    
               fig, ax = plt.subplots()
               sns.barplot(x=top_feats.values, y=top_feats.index, ax=ax, palette="viridis")
               ax.set_title(f"Top 10 Features for {selected_model_name}")
               st.pyplot(fig)
            with st.expander("View Full Classification Report"):
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

        except FileNotFoundError as e:
            st.error(f"Missing required file: {e}")
    else:
        st.error("CSV must contain the target column 'y'.")
else:
    st.info("Upload the CSV to see metrics.")