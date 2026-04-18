import streamlit as st
import pandas as pd

st.set_page_config(page_title="MealSense", layout="centered")

st.title("🍽️ MealSense: Mid-Day Meal Intelligence System")

st.write("Simple tool to estimate meal requirements")

attendance = st.number_input("Enter Attendance", min_value=0)

if attendance > 0:
    recommended = int(attendance * 1.05)

    st.metric("Recommended Meals", recommended)

    if attendance > 80:
        st.warning("⚠️ High demand school")
    else:
        st.success("✅ Normal demand")
