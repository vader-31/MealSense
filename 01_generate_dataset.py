"""
01_generate_dataset.py
======================
Generates a realistic 250-row synthetic dataset for a government
mid-day meal programme covering 20 schools.

Output: data/midday_meal_dataset.csv
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
import random
import os

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
N_ROWS        = 250
N_SCHOOLS     = 20
START_DATE    = date(2024, 6, 1)
DATE_RANGE    = 180   # days of calendar to draw from

# ── Build school profiles (each school has a typical attendance band) ─────────
school_ids = [f"SCH{str(i).zfill(3)}" for i in range(1, N_SCHOOLS + 1)]

school_profiles = {
    sid: {
        "avg_att": random.randint(120, 280),
        "std_att": random.randint(15, 45),
    }
    for sid in school_ids
}

# ── Generate weekday date pool ────────────────────────────────────────────────
all_dates  = [START_DATE + timedelta(days=i) for i in range(DATE_RANGE)]
weekdays   = [d for d in all_dates if d.weekday() < 5]

# ── Generate rows ─────────────────────────────────────────────────────────────
rows = []
for i in range(N_ROWS):
    school_id  = random.choice(school_ids)
    profile    = school_profiles[school_id]

    # Attendance: normally distributed around school average
    attendance = int(np.clip(
        np.random.normal(profile["avg_att"], profile["std_att"]),
        40, 320
    ))

    # Meals served: slightly below attendance (kitchen prepares ~97% of expected)
    meals_served = max(0, attendance + random.randint(-12, 0))

    # Food quality: triangular distribution, skewed toward 7
    food_quality_score = round(random.triangular(3.0, 10.0, 7.0), 1)

    # Complaints: inversely correlated with quality
    base_complaints = max(0, int((10 - food_quality_score) * random.uniform(0, 1.8)))
    complaints_count = max(0, base_complaints + random.randint(-1, 2))

    rows.append({
        "student_id":         f"STU{str(i + 1).zfill(4)}",
        "school_id":          school_id,
        "date":               random.choice(weekdays).strftime("%Y-%m-%d"),
        "attendance":         attendance,
        "meals_served":       meals_served,
        "food_quality_score": food_quality_score,
        "complaints_count":   complaints_count,
    })

# ── Assemble and save ─────────────────────────────────────────────────────────
df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/midday_meal_dataset.csv", index=False)

print("=" * 55)
print("DATASET GENERATED")
print("=" * 55)
print(f"Rows            : {len(df)}")
print(f"Schools         : {df['school_id'].nunique()}")
print(f"Date range      : {df['date'].min()}  →  {df['date'].max()}")
print(f"Avg attendance  : {df['attendance'].mean():.1f}")
print(f"Avg meals served: {df['meals_served'].mean():.1f}")
print(f"\nSaved → data/midday_meal_dataset.csv")
print("\nFirst 5 rows:")
print(df.head().to_string(index=False))
