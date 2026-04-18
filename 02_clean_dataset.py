"""
02_clean_dataset.py
===================
7-step data cleaning pipeline for the mid-day meal dataset.

Steps:
  1. Fix data types
  2. Handle missing values
  3. Remove duplicate rows
  4. Fix logical inconsistency (meals_served > attendance)
  5. Clamp values to valid ranges
  6. Standardise text columns
  7. Sort and reset index

Input : data/midday_meal_dataset.csv
Output: data/midday_meal_cleaned.csv
"""

import pandas as pd
import os

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/midday_meal_dataset.csv")
print(f"Rows loaded: {len(df)}")

# ── STEP 1 — Fix data types ───────────────────────────────────────────────────
# CSV stores everything as text; convert each column to its correct type.

df["date"]               = pd.to_datetime(df["date"])
df["attendance"]         = df["attendance"].astype(int)
df["meals_served"]       = df["meals_served"].astype(int)
df["food_quality_score"] = df["food_quality_score"].astype(float)
df["complaints_count"]   = df["complaints_count"].astype(int)
df["student_id"]         = df["student_id"].astype(str).str.strip()
df["school_id"]          = df["school_id"].astype(str).str.strip()

print("\n✅ STEP 1 — Data types fixed")
print(df.dtypes)

# ── STEP 2 — Handle missing values ───────────────────────────────────────────
# Fill gaps with sensible defaults instead of dropping rows.

missing_before = df.isnull().sum().sum()

df["attendance"]         = df["attendance"].fillna(df["attendance"].median())
df["meals_served"]       = df["meals_served"].fillna(df["meals_served"].median())
df["food_quality_score"] = df["food_quality_score"].fillna(df["food_quality_score"].median())
df["complaints_count"]   = df["complaints_count"].fillna(0)

missing_after = df.isnull().sum().sum()
print(f"\n✅ STEP 2 — Missing values: {missing_before} before → {missing_after} after")

# ── STEP 3 — Remove duplicate rows ───────────────────────────────────────────
# Keep only first occurrence of exact duplicates (data entry errors).

dupes = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"\n✅ STEP 3 — Duplicate rows removed: {dupes}")

# ── STEP 4 — Fix logical inconsistency ───────────────────────────────────────
# meals_served > attendance is impossible; cap at attendance.

inconsistent = (df["meals_served"] > df["attendance"]).sum()
df["meals_served"] = df[["meals_served", "attendance"]].min(axis=1)
print(f"\n✅ STEP 4 — Rows where meals_served > attendance (fixed): {inconsistent}")

# ── STEP 5 — Clamp values to valid ranges ────────────────────────────────────
# Quality score: 1–10. Attendance and complaints: non-negative.

df["food_quality_score"] = df["food_quality_score"].clip(1, 10)
df["attendance"]         = df["attendance"].clip(lower=0)
df["complaints_count"]   = df["complaints_count"].clip(lower=0)
print("\n✅ STEP 5 — Out-of-range values clamped")

# ── STEP 6 — Standardise text columns ────────────────────────────────────────
# Strip whitespace and force uppercase so "sch001" == "SCH001".

df["student_id"] = df["student_id"].str.strip().str.upper()
df["school_id"]  = df["school_id"].str.strip().str.upper()
print("\n✅ STEP 6 — Text columns standardised")

# ── STEP 7 — Sort and reset index ────────────────────────────────────────────

df.sort_values("date", inplace=True)
df.reset_index(drop=True, inplace=True)
print("\n✅ STEP 7 — Sorted by date, index reset")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("CLEANED DATASET SUMMARY")
print("=" * 55)
print(f"Total rows       : {len(df)}")
print(f"Missing values   : {df.isnull().sum().sum()}")
print(f"Date range       : {df['date'].min().date()}  →  {df['date'].max().date()}")
print(f"Schools covered  : {df['school_id'].nunique()}")
print("\nBasic statistics:")
print(df.describe().round(2))

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/midday_meal_cleaned.csv", index=False)
print("\n💾 Saved → data/midday_meal_cleaned.csv")
