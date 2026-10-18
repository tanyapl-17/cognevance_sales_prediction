
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from anomaly_detection import detect_anomalies
from customer_segmentation import segment_customers


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="SalesIQ | Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------- CUSTOM DESIGN --------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1020, #111a30);
        color: #f8fafc;
    }
    [data-testid="stSidebar"] {
        background-color: #10182b;
    }
    [data-testid="stMetric"] {
        background: #17233a;
        padding: 18px;
        border: 1px solid #263753;
        border-radius: 14px;
    }
    h1, h2, h3 {
        color: #f8fafc;
    }
    div[data-testid="stMetricValue"] {
        color: #60a5fa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- LOAD FILES --------------------
DATA_FILE = Path("sales_data.csv")
MODEL_FILE = Path("sales_model.pkl")


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    return data.sort_values("Date").reset_index(drop=True)


@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as file:
        return pickle.load(file)


st.title("📊 SalesIQ")
st.caption("AI-Powered Sales Intelligence Platform")
st.write(
    "Explore sales performance, predict revenue, detect unusual records, "
    "and discover sales profiles."
)

if not DATA_FILE.exists():
    st.error("sales_data.csv was not found in your project folder.")
    st.stop()

if not MODEL_FILE.exists():
    st.error("sales_model.pkl was not found. Run: python train_model.py")
    st.stop()

try:
    data = load_data()
    bundle = load_model()
except Exception as error:
    st.error(f"Could not load the project files: {error}")
    st.stop()


required_columns = [
    "Date", "Product", "Region", "Units_Sold", "Unit_Price",
    "Discount_Percent", "Advertising_Spend", "Revenue", "Cost", "Profit"
]

missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    st.error(f"Missing columns in sales_data.csv: {missing_columns}")
    st.stop()

if data.empty:
    st.error("Your sales dataset is empty.")
    st.stop()


# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.title("⚙️ Control Center")
st.sidebar.caption("Customize your analytics")

min_date = data["Date"].min().date()
max_date = data["Date"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

regions = sorted(data["Region"].dropna().unique().tolist())
products = sorted(data["Product"].dropna().unique().tolist())

selected_regions = st.sidebar.multiselect(
    "Select regions",
    options=regions,
    default=regions,
)

selected_products = st.sidebar.multiselect(
    "Select products",
    options=products,
    default=products,
)

# Always define filtered_data before using it.
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered_data = data[
    (data["Date"].dt.date >= start_date)
    & (data["Date"].dt.date <= end_date)
    & (data["Region"].isin(selected_regions))
    & (data["Product"].isin(selected_products))
].copy()

if filtered_data.empty:
    st.warning("No records match your selected filters. Change the filters.")
    st.stop()


# -------------------- KPI METRICS --------------------
total_revenue = filtered_data["Revenue"].sum()
total_profit = filtered_data["Profit"].sum()
total_records = len(filtered_data)
profit_margin = (
    total_profit / total_revenue * 100 if total_revenue != 0 else 0
)

st.subheader("📌 Business Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
col2.metric("Total Profit", f"₹{total_profit:,.0f}")
col3.metric("Sales Records", f"{total_records:,}")
col4.metric("Profit Margin", f"{profit_margin:.1f}%")


# -------------------- SALES TRENDS --------------------
st.divider()
st.subheader("📈 Sales Performance")

daily_sales = (
    filtered_data.groupby("Date", as_index=False)
    .agg(Revenue=("Revenue", "sum"), Profit=("Profit", "sum"))
)

left, right = st.columns(2)

with left:
    fig = px.line(
        daily_sales,
        x="Date",
        y="Revenue",
        title="Revenue Over Time",
        markers=True,
        template="plotly_dark",
    )
    fig.update_traces(line_color="#60a5fa")
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.line(
        daily_sales,
        x="Date",
        y="Profit",
        title="Profit Over Time",
        markers=True,
        template="plotly_dark",
    )
    fig.update_traces(line_color="#34d399")
    st.plotly_chart(fig, use_container_width=True)


# -------------------- PRODUCT ANALYSIS --------------------
st.divider()
st.subheader("🛍️ Product Intelligence")

product_summary = (
    filtered_data.groupby("Product", as_index=False)
    .agg(
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Units_Sold=("Units_Sold", "sum"),
    )
)

left, right = st.columns(2)

with left:
    fig = px.pie(
        product_summary,
        names="Product",
        values="Revenue",
        title="Revenue Share by Product",
        hole=0.45,
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.bar(
        product_summary.sort_values("Profit", ascending=False),
        x="Product",
        y="Profit",
        color="Profit",
        title="Profit by Product",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(product_summary, use_container_width=True, hide_index=True)


# -------------------- REGIONAL ANALYSIS --------------------
st.divider()
st.subheader("🌍 Regional Performance")

region_summary = (
    filtered_data.groupby("Region", as_index=False)
    .agg(
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
    )
)

fig = px.bar(
    region_summary,
    x="Region",
    y=["Revenue", "Profit"],
    barmode="group",
    title="Revenue and Profit by Region",
    template="plotly_dark",
)
st.plotly_chart(fig, use_container_width=True)


# -------------------- MODEL PERFORMANCE --------------------
st.divider()
st.subheader("🤖 Machine Learning Model")

st.write(f"**Selected model:** {bundle.get('model_name', 'Trained model')}")

metrics = bundle.get("metrics", [])

if metrics:
    st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)

st.caption(
    "MAE and RMSE measure prediction error; lower is generally better. "
    "R² measures how well predictions fit the test data; values closer to 1 "
    "generally indicate a better fit."
)


# -------------------- REVENUE PREDICTION --------------------
st.divider()
st.subheader("🔮 Revenue Prediction")
st.write("Enter sales details to estimate revenue.")

with st.form("revenue_prediction_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        prediction_date = st.date_input(
            "Sales date", value=max_date
        )
        units = st.number_input(
            "Units sold", min_value=1, max_value=100000, value=10
        )
        unit_price = st.number_input(
            "Unit price (₹)", min_value=1.0, value=1000.0, step=100.0
        )

    with c2:
        discount = st.number_input(
            "Discount (%)", min_value=0.0, max_value=100.0,
            value=5.0, step=1.0
        )
        advertising = st.number_input(
            "Advertising spend (₹)", min_value=0.0,
            value=500.0, step=100.0
        )
        prediction_product = st.selectbox("Product", products)

    with c3:
        prediction_region = st.selectbox("Region", regions)
        st.info("The estimate depends on the inputs and the trained model.")

    predict_button = st.form_submit_button("Predict Revenue")

if predict_button:
    feature_values = {
        "Units_Sold": units,
        "Unit_Price": unit_price,
        "Discount_Percent": discount,
        "Advertising_Spend": advertising,
        "DayOfWeek": pd.Timestamp(prediction_date).dayofweek,
        "Month": pd.Timestamp(prediction_date).month,
        "DayOfYear": pd.Timestamp(prediction_date).dayofyear,
        "Product": prediction_product,
        "Region": prediction_region,
    }

    prediction_frame = pd.DataFrame([feature_values])
    prediction_frame = pd.get_dummies(
        prediction_frame,
        columns=["Product", "Region"],
        dtype=int,
    )

    prediction_frame = prediction_frame.reindex(
        columns=bundle["encoded_columns"],
        fill_value=0,
    )

    try:
        predicted_revenue = float(
            bundle["model"].predict(prediction_frame)[0]
        )
        st.success(f"Estimated Revenue: ₹{max(0, predicted_revenue):,.2f}")
        st.caption(
            "This is a model estimate based on synthetic training data, "
            "not a guaranteed business result."
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")


# -------------------- ANOMALY DETECTION --------------------
st.divider()
st.subheader("🚨 Unusual Sales Record Detection")

st.write(
    "Isolation Forest identifies records that differ from typical patterns "
    "in revenue, profit, units sold, and discount."
)

try:
    anomaly_data = detect_anomalies(filtered_data)

    unusual_data = anomaly_data[
        anomaly_data["Anomaly"] == "Unusual"
    ]

    a, b = st.columns(2)
    a.metric("Records Analysed", len(anomaly_data))
    b.metric("Unusual Records Flagged", len(unusual_data))

    if unusual_data.empty:
        st.info(
            "No unusual records were flagged in this selection, "
            "or there were too few valid records to evaluate."
        )
    else:
        st.dataframe(
            unusual_data,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Unusual Records",
            data=unusual_data.to_csv(index=False).encode("utf-8"),
            file_name="unusual_sales_records.csv",
            mime="text/csv",
        )

except Exception as error:
    st.warning(f"Anomaly detection could not run: {error}")


# -------------------- SALES PROFILE SEGMENTATION --------------------
st.divider()
st.subheader("🧩 Sales Profile Segmentation")

st.write(
    "K-Means groups sales records using revenue, profit, units sold, "
    "and discount percentage. These are sales profiles, not individual "
    "customer segments, because the dataset has no customer IDs."
)

try:
    segmented_data = segment_customers(filtered_data)

    assigned_segments = segmented_data[
        segmented_data["Segment"] != "Not assigned"
    ].copy()

    if len(assigned_segments) < 4:
        st.info("Select at least four valid sales records to create segments.")
    else:
        segment_summary = (
            assigned_segments.groupby("Segment", as_index=False)
            .agg(
                Records=("Revenue", "size"),
                Average_Revenue=("Revenue", "mean"),
                Average_Profit=("Profit", "mean"),
                Average_Units=("Units_Sold", "mean"),
                Average_Discount=("Discount_Percent", "mean"),
            )
            .round(2)
        )

        st.dataframe(
            segment_summary,
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            segment_summary,
            x="Segment",
            y="Average_Revenue",
            color="Segment",
            title="Average Revenue by Sales Profile",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download Segmentation Results",
            data=segmented_data.to_csv(index=False).encode("utf-8"),
            file_name="sales_segments.csv",
            mime="text/csv",
        )

except Exception as error:
    st.warning(f"Segmentation could not run: {error}")


# -------------------- RAW DATA --------------------
st.divider()
st.subheader("📋 Explore Sales Data")

st.dataframe(
    filtered_data.sort_values("Date", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download Filtered Sales Data",
    data=filtered_data.to_csv(index=False).encode("utf-8"),
    file_name="filtered_sales_data.csv",
    mime="text/csv",
)


# -------------------- PROFESSIONAL FOOTER --------------------
st.divider()

st.markdown(
    """
    <div style="
        text-align: center;
        padding: 25px 15px;
        background: linear-gradient(135deg, #10182b, #17233a);
        border: 1px solid #263753;
        border-radius: 16px;
        margin-top: 10px;
    ">
        <h2 style="color: #60a5fa; margin-bottom: 8px;">
            📊 SalesIQ
        </h2>

        <p style="color: #cbd5e1; font-size: 15px;">
            AI-Powered Sales Intelligence Platform
        </p>

        <p style="color: #f8fafc; font-size: 16px;">
            Developed by <strong>Tanya Patel</strong>
        </p>

        <p style="color: #60a5fa; font-size: 15px;">
            <strong>Aspiring AI/ML Engineer | Python Developer</strong>
        </p>

        <p style="color: #cbd5e1; font-size: 13px;">
            Passionate about Artificial Intelligence, Machine Learning
            and Python Development.
        </p>

        <p style="color: #94a3b8; font-size: 12px;">
            Python · Pandas · NumPy · Scikit-learn · Streamlit · Plotly
        </p>

        <hr style="border-color: #263753;">

        <p style="color: #94a3b8; font-size: 12px;">
            © 2026 Tanya Patel | Educational Portfolio Project
        </p>

        <p style="color: #94a3b8; font-size: 12px; margin-bottom: 0;">
            Uses synthetic sales data. Validate models before real-world use.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)



