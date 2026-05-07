import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# =========================
# LOAD DATA + MODEL
# =========================
df = pd.read_csv("data/processed.csv")
model = joblib.load("models/lead_time_model.pkl")

# =========================
# RECREATE FEATURES (same as training)
# =========================

# time features
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Order_Year"] = df["Order Date"].dt.year
df["Order_Month"] = df["Order Date"].dt.month
df["Order_Day"] = df["Order Date"].dt.day

# encode categorical columns (same as training)
categorical_cols = ["Ship Mode", "Region", "Factory", "Division"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

factory_codes = df["Factory"].unique().tolist()


# =========================
# SIMULATION FUNCTION
# =========================
def simulate_product(product_name):

    product_rows = df[df["Product Name"] == product_name]

    if product_rows.empty:
        print("Product not found")
        return

    sample = product_rows.iloc[0].copy()

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

    results = []

    # simulate all factories
    for code in factory_codes:

        sample["Factory"] = code

        pred = model.predict(
            sample[features].to_frame().T
        )[0]

        results.append({
            "Factory": code,
            "Predicted_Lead_Time": round(pred, 2)
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("Predicted_Lead_Time")

    print("\n===== Factory Recommendations =====")
    print(result_df)


# =========================
# RUN TEST
# =========================
simulate_product("Laffy Taffy")