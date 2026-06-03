import streamlit as st
import pandas as pd

st.set_page_config(page_title="Model Benchmark Dashboard", layout="wide")

# =======================
# LOAD DATA
# =======================
@st.cache_data
def load_data():
    df1 = pd.read_csv("test_set_comparison_results_final.csv")
    df2 = pd.read_csv("hardware_stress_test_results.csv")
    return df1, df2

df1, df2 = load_data()

st.title("📊 Model Benchmark Dashboard")

tab1, tab2 = st.tabs(["📌 Model Metrics", "⚡ Stress Test Pair Comparison"])

# =======================
# TAB 1 (giữ nguyên)
# =======================
with tab1:
    st.subheader("Test Set Comparison Results")
    st.dataframe(df1, use_container_width=True)

    metric_cols = [c for c in ["mAP50", "Recall", "Precision"] if c in df1.columns]
    if "Model" in df1.columns and metric_cols:
        st.bar_chart(df1.set_index("Model")[metric_cols])

# =======================
# TAB 2 - PAIR COMPARISON
# =======================
with tab2:
    st.subheader("Hardware Stress Test - Paired Comparison")

    df = df2.copy()

    # group keys
    group_keys = ["Image_Count", "Concurrent_Users"]

    # model pivot: mỗi metric thành 2 cột YOLO vs RTDETR
    metric_cols = [
        "Total_Time_Sec",
        "Avg_FPS",
        "Avg_Time_Per_User_Sec",
        "Avg_Time_Per_Image_Sec",
        "Peak_VRAM_MB",
        "GPU_Load_Percent"
    ]

    # chọn model order ổn định
    models = df["Model"].unique()

    selected_metric = st.selectbox("Select Metric", metric_cols)

    # pivot table
    pivot = df.pivot_table(
        index=group_keys,
        columns="Model",
        values=selected_metric
    ).reset_index()

    st.dataframe(pivot, use_container_width=True)