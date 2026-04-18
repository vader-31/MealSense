import streamlit as st
import pandas as pd

st.title("MealSense Dashboard")

df = pd.read_csv("midday_meal_cleaned.csv")

st.subheader("📊 Dataset Overview")
st.write(df.head())

st.subheader("📈 Key Metrics")

st.metric("Avg Attendance", int(df["attendance"].mean()))
st.metric("Avg Meals", int(df["meals_served"].mean()))

st.subheader("🏫 School Analysis")

school = st.selectbox("Select School", df["school_id"].unique())

school_df = df[df["school_id"] == school]

st.write("Avg Attendance:", int(school_df["attendance"].mean()))
st.write("Avg Meals:", int(school_df["meals_served"].mean()))
