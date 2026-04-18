"""
03_explore_dataset.py
=====================
Basic dataset exploration: load, inspect shape, check missing values,
and print summary statistics.

Input: data/midday_meal_cleaned.csv
"""

import pandas as pd

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/midday_meal_cleaned.csv")

# ── First few rows ────────────────────────────────────────────────────────────
print("=" * 55)
print("FIRST 5 ROWS")
print("=" * 55)
print(df.head().to_string())

# ── Shape ─────────────────────────────────────────────────────────────────────
print(f"\nDataset has {df.shape[0]} rows and {df.shape[1]} columns.")

# ── Missing values ────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("MISSING VALUES PER COLUMN")
print("=" * 55)
missing = df.isnull().sum()
print(missing)
if missing.sum() == 0:
    print("\n✅ No missing values — dataset is clean.")
else:
    print(f"\n⚠️  Total missing: {missing.sum()}")

# ── Basic statistics ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("BASIC STATISTICS (numeric columns)")
print("=" * 55)
print(df.describe().round(2))

# ── Unique counts ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("UNIQUE VALUE COUNTS")
print("=" * 55)
print(f"Unique students  : {df['student_id'].nunique()}")
print(f"Unique schools   : {df['school_id'].nunique()}")
print(f"Date range       : {df['date'].min()}  →  {df['date'].max()}")
