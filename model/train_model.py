import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import os

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('model', exist_ok=True)

# Load data
df = pd.read_csv('data/bank-full.csv', sep=';') 

# Map target 'y' to 1/0 - One-Hot Encoding Target
df['y'] = df['y'].str.strip().map({'yes': 1, 'no': 0})
df = df.dropna(subset=['y'])

# SPLIT DATA FIRST
# We split before encoding so 'test.csv' stays readable (17 columns)
train_bef, test_bef = train_test_split(df, test_size=0.2, random_state=42)

# 3. Convert text to numbers
df_encoded = pd.get_dummies(df)

# 4. Define X and y
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']

# Save column names (CRITICAL for Streamlit consistency)
joblib.dump(X.columns.tolist(), 'model/model_columns.pkl')

# Re-split the encoded data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Save CSVs
test_bef.to_csv('data/test.csv', index=False)
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)
train_df.to_csv('data/train.csv', index=False)

# 6. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Save the scaler so the app uses the same math
joblib.dump(scaler, 'model/scaler.pkl')

# 7. Train and Save models
models_dict = {
    "logistic_reg": LogisticRegression(
        max_iter=2000, 
        C=1.0, 
        class_weight='balanced', 
        random_state=42
    ),
    "decision_tree": DecisionTreeClassifier(
        max_depth=15, 
        min_samples_split=10, 
        random_state=42
    ),
    "knn": KNeighborsClassifier(
        n_neighbors=5, 
        weights='distance', 
        metric='euclidean'
    ),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        n_jobs=-1,        # Uses all CPU cores for faster training
        random_state=42   # Ensures consistent results every run
    ),
    "xgboost": xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
}

for name, m in models_dict.items():
    m.fit(X_train_scaled, y_train)
    file_path = os.path.join('model', f"{name}.pkl")
    joblib.dump(m, file_path, compress=3)
    print(f"Successfully saved: {file_path}")

print("✨ Training complete! All files saved in 'model/' and 'data/'.")