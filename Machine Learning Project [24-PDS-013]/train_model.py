import pandas as pd
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


file_path = r"C:\Users\Hp\OneDrive\Desktop\Machine Learning Project [24-PDS-013]\data_airline_reviews.csv"
df = pd.read_csv(file_path)


important_features = ["airline", "traveller_type", "cabin", "seat_comfort", 
                      "cabin_service", "food_bev", "entertainment", "value_for_money"]


if "recommended" not in df.columns:
    raise ValueError(" 'recommended' column is missing in the dataset!")


df["recommended"] = df["recommended"].astype(str).str.strip().str.lower()
df["recommended"] = df["recommended"].map({"yes": 1, "no": 0})
df.dropna(subset=["recommended"], inplace=True)
df["recommended"] = df["recommended"].astype(int)


df.fillna({"airline": "Unknown", "traveller_type": "Unknown", "cabin": "Unknown"}, inplace=True)


label_encoders = {}
for col in ["airline", "traveller_type", "cabin"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

    
    le.classes_ = np.append(le.classes_, "Unknown")

    label_encoders[col] = le


joblib.dump(label_encoders, r"C:\Users\Hp\OneDrive\Desktop\Machine Learning Project [24-PDS-013]\label_encoders.pkl")


X = df[important_features]
y = df["recommended"]


if X.isnull().sum().sum() > 0:
    print("\nMissing values found in X! Fixing them...")
    X.fillna(X.mean(), inplace=True)  
    print("\n Missing values fixed!")


print("\n🔍 Missing values in X after cleaning:\n", X.isnull().sum())


print("\n🔍 Checking feature data (X):", X.shape)


if X.shape[0] == 0:
    raise ValueError(" X is empty! Please check data preprocessing.")


print("\n Before SMOTE:\n", y.value_counts())


smote = SMOTE(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


print("\n After SMOTE:\n", pd.Series(y_resampled).value_counts())


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


scaler_path = r"C:\Users\Hp\OneDrive\Desktop\Machine Learning Project [24-PDS-013]\scaler.pkl"
joblib.dump(scaler, scaler_path)


model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
model.fit(X_train_scaled, y_train)


y_train_pred = model.predict(X_train_scaled)
print("\n Training Set Predictions:\n", pd.Series(y_train_pred).value_counts())


model_path = r"C:\Users\Hp\OneDrive\Desktop\Machine Learning Project [24-PDS-013]\rf_model.pkl"
joblib.dump(model, model_path)

print(f"\n Model trained and saved successfully at {model_path}!")

