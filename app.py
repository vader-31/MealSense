import streamlit as st
import pandas as pd

# ---------------- CONFIG ----------------
st.set_page_config(page_title="MealSense Dashboard", layout="wide")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("midday_meal_cleaned.csv")

# ---------------- TITLE ----------------
st.title("🍽️ MealSense Dashboard")
st.markdown("Optimize and monitor school meal distribution using data-driven insights")

st.markdown("---")

# ---------------- TOP METRICS ----------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", len(df))
col2.metric("Avg Attendance", int(df["attendance"].mean()))
col3.metric("Avg Meals Served", int(df["meals_served"].mean()))
col4.metric("Total Schools", df["school_id"].nunique())

st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔍 Filters")

selected_school = st.sidebar.selectbox(
    "Select School ID", sorted(df["school_id"].unique())
)

min_attendance = st.sidebar.slider(
    "Minimum Attendance", 0, int(df["attendance"].max()), 0
)

# ---------------- FILTER DATA ----------------
filtered_df = df[
    (df["school_id"] == selected_school) &
    (df["attendance"] >= min_attendance)
]

# ---------------- RECOMMENDATION ----------------
st.subheader("📊 Meal Recommendation Tool")

attendance_input = st.number_input("Enter Today's Attendance", min_value=0)

if attendance_input > 0:
    recommended = int(attendance_input * 1.05)

    colA, colB = st.columns(2)

    colA.metric("Recommended Meals", recommended)
    colB.metric("Buffer", recommended - attendance_input)

    if attendance_input > 80:
        st.error("⚠️ High demand school - risk of shortage")
    else:
        st.success("✅ Normal demand")

st.markdown("---")

# ---------------- SCHOOL INSIGHTS ----------------
st.subheader(f"🏫 Insights for School {selected_school}")

avg_att = int(filtered_df["attendance"].mean())
avg_meal = int(filtered_df["meals_served"].mean())

colX, colY = st.columns(2)

colX.metric("Avg Attendance", avg_att)
colY.metric("Avg Meals Served", avg_meal)

if avg_meal < avg_att:
    st.warning("⚠️ Under-supply detected")
else:
    st.success("✅ Supply sufficient")

# ---------------- CHART ----------------
st.subheader("📈 Meals Distribution Across Schools")

chart_data = df.groupby("school_id")["meals_served"].mean()

st.bar_chart(chart_data, use_container_width=True)

# ---------------- DATA PREVIEW ----------------
st.subheader("📋 Data Preview")

st.dataframe(filtered_df.head(10), use_container_width=True)

# ---------------- DOWNLOAD OPTION ----------------
st.subheader("⬇️ Download Filtered Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="filtered_meal_data.csv",
    mime="text/csv"
)

# ---------------- SYSTEM INSIGHTS ----------------
st.subheader("📌 System Insights")

shortage_days = (df["meals_served"] < df["attendance"]).sum()
avg_gap = round((df["meals_served"] - df["attendance"]).mean(), 2)

st.write(f"🔴 Total shortage days: {shortage_days}")
st.write(f"📉 Average gap: {avg_gap}")

st.markdown("---")
st.caption("Built by Samdarshi | Data Science Project")
