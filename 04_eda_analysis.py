"""
04_eda_analysis.py
==================
Deep exploratory data analysis:
  - Average meals served per school
  - Average attendance per school
  - Complaints distribution
  - Correlation matrix with interpretation

Saves: outputs/eda_correlation_heatmap.png
       outputs/eda_complaints_distribution.png
       outputs/eda_school_averages.png

Input: data/midday_meal_cleaned.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/midday_meal_cleaned.csv")
print(f"Rows loaded: {len(df)}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — Average meals served per school
# ═══════════════════════════════════════════════════════════════════════════════

avg_meals = (
    df.groupby("school_id")["meals_served"]
    .mean().round(2).sort_values(ascending=False).reset_index()
)
avg_meals.columns = ["school_id", "avg_meals_served"]

print("=" * 50)
print("AVERAGE MEALS SERVED PER SCHOOL")
print("=" * 50)
print(avg_meals.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Average attendance per school
# ═══════════════════════════════════════════════════════════════════════════════

avg_attendance = (
    df.groupby("school_id")["attendance"]
    .mean().round(2).sort_values(ascending=False).reset_index()
)
avg_attendance.columns = ["school_id", "avg_attendance"]

print("\n" + "=" * 50)
print("AVERAGE ATTENDANCE PER SCHOOL")
print("=" * 50)
print(avg_attendance.to_string(index=False))

# ── Chart: school averages ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(avg_meals))
ax.barh([s for s in avg_meals["school_id"]], avg_meals["avg_meals_served"],
        color="#378ADD", alpha=0.85, label="Avg meals served")
ax.barh([s for s in avg_attendance["school_id"]],
        avg_attendance.set_index("school_id").loc[avg_meals["school_id"], "avg_attendance"].values,
        color="#D3D1C7", alpha=0.6, label="Avg attendance")
ax.set_xlabel("Count")
ax.set_title("Average meals served vs attendance per school")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/eda_school_averages.png", dpi=150)
plt.close()
print("\n📊 Saved → outputs/eda_school_averages.png")

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Complaints distribution
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("COMPLAINTS DISTRIBUTION")
print("=" * 50)
print(df["complaints_count"].describe().round(2))
print("\nDays per complaint level:")
print(df["complaints_count"].value_counts().sort_index().to_string())

fig, ax = plt.subplots(figsize=(9, 5))
vals = df["complaints_count"].value_counts().sort_index()
colors = ["#27500A" if v <= 3 else "#BA7517" if v <= 6 else "#A32D2D" for v in vals.index]
ax.bar(vals.index, vals.values, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Complaints on a given day")
ax.set_ylabel("Number of days")
ax.set_title("Complaints distribution across all school-days")
ax.set_xticks(vals.index)
plt.tight_layout()
plt.savefig("outputs/eda_complaints_distribution.png", dpi=150)
plt.close()
print("\n📊 Saved → outputs/eda_complaints_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4 — Correlation matrix
# ═══════════════════════════════════════════════════════════════════════════════

cols = ["attendance", "meals_served", "food_quality_score", "complaints_count"]
corr = df[cols].corr().round(3)

print("\n" + "=" * 50)
print("CORRELATION MATRIX")
print("=" * 50)
print(corr.to_string())

fig, ax = plt.subplots(figsize=(7, 5))
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(
    corr, annot=True, fmt=".3f", cmap="RdYlGn",
    center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8}
)
ax.set_title("Correlation matrix — numeric columns")
plt.tight_layout()
plt.savefig("outputs/eda_correlation_heatmap.png", dpi=150)
plt.close()
print("\n📊 Saved → outputs/eda_correlation_heatmap.png")

# ── Key correlation insights ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("KEY CORRELATION INSIGHTS")
print("=" * 50)

pairs = [
    (cols[i], cols[j], corr.iloc[i, j])
    for i in range(len(cols))
    for j in range(i + 1, len(cols))
]
pairs.sort(key=lambda x: abs(x[2]), reverse=True)

for col1, col2, val in pairs:
    strength  = "strong" if abs(val) >= 0.4 else "moderate" if abs(val) >= 0.2 else "weak / no"
    direction = "positive" if val > 0 else "negative"
    print(f"  {col1} vs {col2}: {val:+.3f}  →  {strength} {direction} relationship")
