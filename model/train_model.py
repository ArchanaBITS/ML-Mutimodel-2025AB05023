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

# Load your dataset (Ensure it has >12 features and >500 rows)
df = pd.read_csv('data/heart.csv') 
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Define the 6 models
model = {
    "logistic_reg": LogisticRegression(),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(),
    "xgboost": xgb.XGBClassifier()
}
model_dir = 'model'
# Train and Save each model
for name, model in model.items():
    model.fit(X_train, y_train)
# Define the save path inside the 'model' folder
    file_path = os.path.join(model_dir, f"{name}.pkl")
    
    # Save the model
    joblib.dump(model, file_path)
    print(f"Successfully saved: {file_path}")