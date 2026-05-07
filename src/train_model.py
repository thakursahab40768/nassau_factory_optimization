import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load processed data
df = pd.read_csv("data/processed.csv")
# Convert dates
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])

# Time features
df["Order_Year"] = df["Order Date"].dt.year
df["Order_Month"] = df["Order Date"].dt.month
df["Order_Day"] = df["Order Date"].dt.day

print(df.shape)
print(df.head())
#we will predict Lead Time Months
features = [
    "Distance_km",
    "Units",
    "Sales",
    "Cost",
    "Gross Profit",
    "Ship Mode",
    "Region",
    "Factory",
    "Division",
    "Order_Year",
    "Order_Month",
    "Order_Day"
]

target = "Lead Time Months"

categorical_cols = ["Ship Mode", "Region", "Factory", "Division"]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE:", rmse)
print("R2:", r2_score(y_test, preds))

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

print("\n===== Random Forest =====")
print("MAE:", mean_absolute_error(y_test, rf_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds)))
print("R2:", r2_score(y_test, rf_preds))

import joblib

# Save Random Forest model
joblib.dump(rf_model, "models/lead_time_model.pkl")

print("\nModel saved successfully!")