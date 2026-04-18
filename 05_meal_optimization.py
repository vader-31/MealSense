"""
05_meal_optimization.py
=======================
Supply anomaly detection and meal allocation optimisation.

Steps:
  1. Estimate required and recommended meals (attendance + 5% buffer)
  2. Detect over-supply / under-supply / optimal days
  3. Per-school anomaly report
  4. Suggested daily allocation ranked by urgency
  5. Top 10 worst under-supply days
  6. System-wide efficiency summary

Saves: data/midday_meal_optimised.csv
       data/school_supply_report.csv
       outputs/opt_supply_status.png
       outputs/opt_school_deficit.png

Input: data/midday_meal_cleaned.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

BUFFER = 0.05   # 5% safety margin

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/midday_meal_cleaned.csv")
print(f"Rows loaded: {len(df)}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Estimate required meals
#   required_meals    = attendance          (bare minimum)
#   recommended_meals = attendance × 1.05  (with safety buffer)
# ═══════════════════════════════════════════════════════════════════════════════

df["required_meals"]    = df["attendance"]
df["recommended_meals"] = (df["attendance"] * (1 + BUFFER)).astype(int)

print("=" * 55)
print("STEP 1 — MEAL ESTIMATES")
print("=" * 55)
print(f"Buffer used           : {int(BUFFER * 100)}%")
print(f"Avg attendance        : {df['attendance'].mean():.1f}")
print(f"Avg required meals    : {df['required_meals'].mean():.1f}")
print(f"Avg recommended meals : {df['recommended_meals'].mean():.1f}")
print(f"Avg meals served      : {df['meals_served'].mean():.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Detect over-supply / under-supply / optimal
# ═══════════════════════════════════════════════════════════════════════════════

def classify_supply(row):
    if row["meals_served"] > row["recommended_meals"]:
        return "over-supply"
    elif row["meals_served"] < row["required_meals"]:
        return "under-supply"
    return "optimal"

df["supply_status"] = df.apply(classify_supply, axis=1)
df["gap"]           = df["meals_served"] - df["recommended_meals"]

status_counts = df["supply_status"].value_counts()

print("\n" + "=" * 55)
print("STEP 2 — SUPPLY STATUS")
print("=" * 55)
print(f"  Optimal days      : {status_counts.get('optimal', 0)}")
print(f"  Under-supply days : {status_counts.get('under-supply', 0)}")
print(f"  Over-supply days  : {status_counts.get('over-supply', 0)}")

total_shortage = df.loc[df["gap"] < 0, "gap"].abs().sum()
total_excess   = df.loc[df["gap"] > 0, "gap"].sum()
print(f"\n  Total meals short  : {int(total_shortage)}")
print(f"  Total excess meals : {int(total_excess)}")
print(f"  Avg daily gap      : {df['gap'].mean():.1f} meals")

# ── Chart: supply status pie ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
color_map = {"optimal": "#639922", "under-supply": "#E24B4A", "over-supply": "#BA7517"}
sc = status_counts.reindex(["optimal", "under-supply", "over-supply"]).dropna()
ax.pie(sc.values, labels=sc.index, autopct="%1.1f%%",
       colors=[color_map[k] for k in sc.index], startangle=90)
ax.set_title("Supply status across all school-days")
plt.tight_layout()
plt.savefig("outputs/opt_supply_status.png", dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Per-school anomaly report
# ═══════════════════════════════════════════════════════════════════════════════

df["daily_deficit"] = (df["recommended_meals"] - df["meals_served"]).clip(lower=0)
df["daily_excess"]  = (df["meals_served"] - df["recommended_meals"]).clip(lower=0)

school_report = (
    df.groupby("school_id")
    .agg(
        avg_attendance    = ("attendance",        "mean"),
        avg_meals_served  = ("meals_served",      "mean"),
        avg_recommended   = ("recommended_meals", "mean"),
        avg_daily_deficit = ("daily_deficit",     "mean"),
        avg_daily_excess  = ("daily_excess",      "mean"),
        under_supply_days = ("supply_status",     lambda x: (x == "under-supply").sum()),
        over_supply_days  = ("supply_status",     lambda x: (x == "over-supply").sum()),
        optimal_days      = ("supply_status",     lambda x: (x == "optimal").sum()),
    )
    .round(2)
    .reset_index()
)

print("\n" + "=" * 55)
print("STEP 3 — PER-SCHOOL ANOMALY REPORT")
print("=" * 55)
print(school_report.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Suggested daily allocation per school
# ═══════════════════════════════════════════════════════════════════════════════

allocation = school_report[[
    "school_id", "avg_attendance", "avg_meals_served",
    "avg_recommended", "avg_daily_deficit"
]].copy()
allocation["suggested_daily_allocation"] = (allocation["avg_attendance"] * (1 + BUFFER)).round().astype(int)
allocation["meals_to_add_per_day"]       = (allocation["suggested_daily_allocation"] - allocation["avg_meals_served"]).round(1)
allocation = allocation.sort_values("avg_daily_deficit", ascending=False).reset_index(drop=True)
allocation.index += 1

print("\n" + "=" * 55)
print("STEP 4 — SUGGESTED DAILY ALLOCATION (ranked by urgency)")
print("=" * 55)
print(allocation.to_string())

# ── Chart: per-school deficit ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(allocation["school_id"], allocation["avg_daily_deficit"], color="#E24B4A", alpha=0.85)
ax.set_xlabel("School")
ax.set_ylabel("Avg daily deficit (meals)")
ax.set_title("Average daily meal deficit per school")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("outputs/opt_school_deficit.png", dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Top 10 worst under-supply days
# ═══════════════════════════════════════════════════════════════════════════════

worst_days = (
    df[df["supply_status"] == "under-supply"]
    [["date", "school_id", "attendance", "meals_served",
      "recommended_meals", "gap", "food_quality_score", "complaints_count"]]
    .sort_values("gap")
    .head(10)
    .reset_index(drop=True)
)
worst_days.index += 1

print("\n" + "=" * 55)
print("STEP 5 — TOP 10 WORST UNDER-SUPPLY DAYS")
print("=" * 55)
print(worst_days.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Efficiency summary
# ═══════════════════════════════════════════════════════════════════════════════

current_total     = df["meals_served"].sum()
total_recommended = df["recommended_meals"].sum()
efficiency        = round(current_total / total_recommended * 100, 1)
gap_to_close      = total_recommended - current_total

print("\n" + "=" * 55)
print("STEP 6 — EFFICIENCY SUMMARY")
print("=" * 55)
print(f"Total meals currently served  : {current_total:,}")
print(f"Total meals recommended       : {total_recommended:,}")
print(f"Supply efficiency             : {efficiency}%")
print(f"Meals gap to close            : {gap_to_close:,}")
print(f"Avg meals to add / school/day : {gap_to_close / df['school_id'].nunique() / (len(df) / df['school_id'].nunique()):.1f}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv("data/midday_meal_optimised.csv", index=False)
school_report.to_csv("data/school_supply_report.csv", index=False)

print("\n💾 Saved → data/midday_meal_optimised.csv")
print("💾 Saved → data/school_supply_report.csv")
print("📊 Saved → outputs/opt_supply_status.png")
print("📊 Saved → outputs/opt_school_deficit.png")
