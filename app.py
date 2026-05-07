import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# =========================
# LOAD DATA + MODEL
# =========================
df = pd.read_csv("data/processed.csv")
model = joblib.load("models/lead_time_model.pkl")

# recreate features
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Order_Year"] = df["Order Date"].dt.year
df["Order_Month"] = df["Order Date"].dt.month
df["Order_Day"] = df["Order Date"].dt.day

# encode categorical columns
categorical_cols = ["Ship Mode", "Region", "Factory", "Division"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

factory_codes = df["Factory"].unique().tolist()
# map encoded factory codes → real names
factory_name_map = {
    0: "Lot's O' Nuts",
    1: "Wicked Choccy's",
    2: "Sugar Shack",
    3: "Secret Factory",
    4: "The Other Factory"
}

# =========================
# STREAMLIT UI
# =========================
st.title("🏭 Factory Reallocation & Shipping Optimization")

st.write("Select a product to simulate factory performance.")

product_list = df["Product Name"].unique()

selected_product = st.selectbox("Choose Product", product_list)


def simulate_product(product_name):

    product_rows = df[df["Product Name"] == product_name]

    if product_rows.empty:
        return None

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

    for code in factory_codes:

        sample["Factory"] = code

        pred = model.predict(
            sample[features].to_frame().T
        )[0]

        results.append({
            "Factory": factory_name_map.get(code, code),
            "Predicted Lead Time (Months)": round(pred, 2)
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("Predicted Lead Time (Months)")

    return result_df


# =========================
# BUTTON
# =========================
if st.button("Run Optimization"):

    results = simulate_product(selected_product)

    st.subheader("📊 Factory Recommendations")
    st.dataframe(results)

    st.bar_chart(
        results.set_index("Factory")["Predicted Lead Time (Months)"]
    )

    # BEST FACTORY (MUST BE INSIDE BUTTON)
    best_factory = results.iloc[0]["Factory"]
    best_time = results.iloc[0]["Predicted Lead Time (Months)"]

    st.success(
        f"⭐ Recommended Factory: {best_factory} "
        f"(Predicted Lead Time: {best_time} months)"
    )