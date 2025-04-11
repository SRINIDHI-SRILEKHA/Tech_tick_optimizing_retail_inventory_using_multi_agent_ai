import streamlit as st
import pandas as pd

st.title("📈 Sales Data Viewer")

# 👉 CSV upload
file = st.file_uploader("Upload your sales CSV file", type=["csv"])

if file is not None:
    # ✅ Read the uploaded CSV
    df = pd.read_csv(file)

    # 👀 Show first few rows
    st.subheader("📋 Preview of Uploaded Data")
    st.write(df.head())

    # 📊 Optional chart (if needed)
    if "Predicted Demand" in df.columns:
        st.subheader("📊 Predicted Demand Chart")
        st.bar_chart(df["Predicted Demand"].value_counts().sort_index())
