import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("liver.csv")
df.columns = df.columns.str.strip()

# Encode categorical data
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])  # Male=1, Female=0
df['Target'] = df['Target'].map({1: 1, 2: 0})  # 1: Liver Disease, 0: No Disease

if df['Target'].isnull().any():
    raise ValueError("🚨 Invalid values in Target column after mapping!")

X = df.drop("Target", axis=1)
y = df["Target"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create models folder
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# Train models
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "models/rf_model.pkl")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
joblib.dump(xgb, "models/xgb_model.pkl")

lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train, y_train)
joblib.dump(lgbm, "models/lgbm_model.pkl")

# Evaluate
print("\n📊 Model Evaluation Results:")
for model, name in zip([rf, xgb, lgbm], ['Random Forest', 'XGBoost', 'LightGBM']):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ {name} Accuracy: {acc:.2f}")

print("💾 All models and scaler saved successfully.")
