"""
06_health_scoring.py
====================
Advanced school nutrition analytics:

  1. Calorie precision   — actual vs target calories per school per day
  2. Health scoring      — composite score from 5 nutrition dimensions
  3. Habit tracking      — 28-day service consistency & streaks
  4. Personalised recs   — school-specific actions ranked by impact

Saves: outputs/health_scores.png
       outputs/calorie_precision.png
       outputs/habit_heatmap.png
       outputs/health_report.csv

Input: data/midday_meal_optimised.csv  (from 05_meal_optimization.py)
       data/school_supply_report.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

os.makedirs("outputs", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df     = pd.read_csv("data/midday_meal_optimised.csv")
report = pd.read_csv("data/school_supply_report.csv")

print(f"Rows loaded: {len(df)}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — CALORIE PRECISION
#
# Government ICDS target: 650 kcal for primary, 700 kcal for upper primary.
# We estimate calories from meals_served using a per-meal calorie proxy.
#
# Calorie proxy (simplified):
#   Rice + dal + sabzi meal ≈ 580–720 kcal depending on portion & menu.
#   We model: cal_per_meal = 600 + food_quality_score * 8
#   (higher quality typically means richer ingredients)
# ═══════════════════════════════════════════════════════════════════════════════

CAL_TARGET_PRIMARY       = 650    # kcal per student per day
CAL_TARGET_UPPER_PRIMARY = 700

df["cal_per_meal"]     = 600 + df["food_quality_score"] * 8
df["total_calories"]   = (df["meals_served"] * df["cal_per_meal"]).round(0).astype(int)
df["cal_per_student"]  = np.where(
    df["meals_served"] > 0,
    (df["total_calories"] / df["meals_served"]).round(1),
    0
)
df["cal_target"]       = CAL_TARGET_PRIMARY
df["cal_gap"]          = df["cal_per_student"] - df["cal_target"]
df["cal_precision_pct"]= (df["cal_per_student"] / df["cal_target"] * 100).round(1)

cal_summary = (
    df.groupby("school_id")
    .agg(
        avg_cal_per_student = ("cal_per_student", "mean"),
        avg_cal_gap         = ("cal_gap",          "mean"),
        avg_precision_pct   = ("cal_precision_pct","mean"),
        days_on_target      = ("cal_gap",          lambda x: (x.abs() <= 30).sum()),
        total_days          = ("cal_gap",           "count"),
    )
    .round(2)
    .reset_index()
)
cal_summary["on_target_pct"] = (cal_summary["days_on_target"] / cal_summary["total_days"] * 100).round(1)

print("=" * 55)
print("STEP 1 — CALORIE PRECISION PER SCHOOL")
print("=" * 55)
print(f"System target     : {CAL_TARGET_PRIMARY} kcal/student/day")
print(f"System avg actual : {df['cal_per_student'].mean():.1f} kcal")
print(f"Avg calorie gap   : {df['cal_gap'].mean():.1f} kcal")
print(f"Days within ±30   : {(df['cal_gap'].abs() <= 30).sum()} / {len(df)}")
print("\nPer-school calorie summary:")
print(cal_summary.to_string(index=False))

# ── Chart: calorie precision ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
colors = cal_summary["avg_cal_gap"].apply(
    lambda g: "#639922" if abs(g) <= 30 else "#BA7517" if abs(g) <= 60 else "#E24B4A"
)
ax.bar(cal_summary["school_id"], cal_summary["avg_cal_per_student"],
       color=colors.values, edgecolor="white", linewidth=0.5)
ax.axhline(CAL_TARGET_PRIMARY, color="#185FA5", linestyle="--", linewidth=1.5, label=f"Target {CAL_TARGET_PRIMARY} kcal")
ax.set_xlabel("School")
ax.set_ylabel("Avg calories per student")
ax.set_title("Calorie precision — avg calories per student vs target")
ax.tick_params(axis="x", rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("outputs/calorie_precision.png", dpi=150)
plt.close()
print("\n📊 Saved → outputs/calorie_precision.png")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — HEALTH SCORING
#
# Composite score (0–100) built from 5 weighted dimensions:
#
#   Dimension          Weight   Proxy column
#   ─────────────────────────────────────────
#   food quality         30%    food_quality_score (scaled 1–10 → 0–100)
#   calorie precision    25%    closeness to 650 kcal target
#   supply efficiency    20%    meals_served / recommended_meals
#   complaint load       15%    inverse of complaints_count
#   service consistency  10%    meals served > 0 (binary)
# ═══════════════════════════════════════════════════════════════════════════════

W_QUALITY     = 0.30
W_CALORIES    = 0.25
W_EFFICIENCY  = 0.20
W_COMPLAINTS  = 0.15
W_CONSISTENCY = 0.10

def health_score_row(row):
    quality_score  = (row["food_quality_score"] - 1) / 9 * 100
    cal_score      = max(0, 100 - abs(row["cal_gap"]) / row["cal_target"] * 200)
    eff_score      = min(100, row["meals_served"] / row["recommended_meals"] * 100)
    comp_score     = max(0, 100 - row["complaints_count"] * 10)
    cons_score     = 100 if row["meals_served"] > 0 else 0
    return round(
        W_QUALITY     * quality_score  +
        W_CALORIES    * cal_score      +
        W_EFFICIENCY  * eff_score      +
        W_COMPLAINTS  * comp_score     +
        W_CONSISTENCY * cons_score,
        1
    )

df["health_score"] = df.apply(health_score_row, axis=1)

school_health = (
    df.groupby("school_id")
    .agg(
        avg_health_score  = ("health_score",       "mean"),
        avg_quality       = ("food_quality_score",  "mean"),
        avg_efficiency    = ("cal_precision_pct",   "mean"),
        avg_complaints    = ("complaints_count",    "mean"),
    )
    .round(2)
    .reset_index()
    .sort_values("avg_health_score", ascending=False)
)

def grade(score):
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "F"

school_health["grade"] = school_health["avg_health_score"].apply(grade)

print("\n" + "=" * 55)
print("STEP 2 — HEALTH SCORES PER SCHOOL")
print("=" * 55)
print(school_health.to_string(index=False))

# ── Chart: health scores ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
palette = school_health["avg_health_score"].apply(
    lambda s: "#639922" if s >= 75 else "#BA7517" if s >= 55 else "#E24B4A"
)
bars = ax.bar(school_health["school_id"], school_health["avg_health_score"],
              color=palette.values, edgecolor="white")
ax.axhline(75, color="#185FA5", linestyle="--", linewidth=1, label="Good threshold (75)")
ax.axhline(55, color="#BA7517", linestyle=":",  linewidth=1, label="Warning threshold (55)")
for bar, grade_val in zip(bars, school_health["grade"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            grade_val, ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xlabel("School")
ax.set_ylabel("Health score (0–100)")
ax.set_title("Composite health score per school (A–F graded)")
ax.set_ylim(0, 105)
ax.tick_params(axis="x", rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("outputs/health_scores.png", dpi=150)
plt.close()
print("\n📊 Saved → outputs/health_scores.png")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — HABIT TRACKING
#
# For each school, build a 28-record window of most recent observations.
# A "good habit day" = meals_served >= required_meals AND quality >= 6.
# Metrics: habit rate, current streak, longest streak.
# ═══════════════════════════════════════════════════════════════════════════════

def streak(series):
    """Current streak (from end) and longest streak in a boolean series."""
    vals = list(series)
    cur = 0
    for v in reversed(vals):
        if v: cur += 1
        else: break
    longest, run = 0, 0
    for v in vals:
        run = run + 1 if v else 0
        longest = max(longest, run)
    return cur, longest

habit_records = []
for school, grp in df.groupby("school_id"):
    grp = grp.sort_values("date").tail(28)
    good = (grp["meals_served"] >= grp["required_meals"]) & (grp["food_quality_score"] >= 6)
    cur_streak, max_streak = streak(good)
    habit_records.append({
        "school_id":      school,
        "days_tracked":   len(grp),
        "good_days":      int(good.sum()),
        "habit_rate_pct": round(good.mean() * 100, 1),
        "current_streak": cur_streak,
        "longest_streak": max_streak,
    })

habit_df = pd.DataFrame(habit_records).sort_values("habit_rate_pct", ascending=False)

print("\n" + "=" * 55)
print("STEP 3 — HABIT TRACKING (last 28 days per school)")
print("=" * 55)
print(habit_df.to_string(index=False))

# ── Chart: habit heatmap ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
pivot = df.copy()
pivot["date_ord"] = pd.to_datetime(pivot["date"]).dt.strftime("%Y-%m-%d")
pivot["good_day"] = ((pivot["meals_served"] >= pivot["required_meals"]) & (pivot["food_quality_score"] >= 6)).astype(int)

top_schools = school_health["school_id"].head(10).tolist()
pivot_filt = pivot[pivot["school_id"].isin(top_schools)]
hmap = pivot_filt.pivot_table(index="school_id", columns="date_ord", values="good_day", aggfunc="max")
hmap = hmap.fillna(0)

cmap = mcolors.ListedColormap(["#FEE2E2", "#D1FAE5"])
ax.imshow(hmap.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
ax.set_yticks(range(len(hmap.index)))
ax.set_yticklabels(hmap.index, fontsize=9)
n_dates = len(hmap.columns)
step = max(1, n_dates // 8)
ax.set_xticks(range(0, n_dates, step))
ax.set_xticklabels([hmap.columns[i] for i in range(0, n_dates, step)], rotation=45, fontsize=8)
ax.set_title("Habit heatmap — green = good day (top 10 schools)")
plt.tight_layout()
plt.savefig("outputs/habit_heatmap.png", dpi=150)
plt.close()
print("\n📊 Saved → outputs/habit_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — PERSONALISED RECOMMENDATIONS
#
# Rule-based engine: each school gets ranked actions based on its worst metric.
# Priority: fix critical issues first (score < 55), then improve borderline.
# ═══════════════════════════════════════════════════════════════════════════════

def personalised_recs(row):
    recs = []
    if row["avg_health_score"] < 55:
        recs.append("URGENT: Request district nutrition officer review — score below 55")
    if row["avg_quality"] < 6.0:
        recs.append(f"Improve food quality (current {row['avg_quality']:.1f}/10): add protein dish 3×/week")
    if row["avg_complaints"] > 4:
        recs.append(f"Complaints high ({row['avg_complaints']:.1f}/day): review menu variety and ingredient quality")
    if row["avg_health_score"] >= 80:
        recs.append("Benchmark school: share menu rotation as district template")
    if not recs:
        recs.append("Maintain current performance; pilot student feedback form weekly")
    return " | ".join(recs)

merged = school_health.merge(
    cal_summary[["school_id", "avg_cal_gap", "on_target_pct"]],
    on="school_id", how="left"
).merge(
    habit_df[["school_id", "habit_rate_pct", "current_streak"]],
    on="school_id", how="left"
)
merged["recommendation"] = merged.apply(personalised_recs, axis=1)

print("\n" + "=" * 55)
print("STEP 4 — PERSONALISED RECOMMENDATIONS")
print("=" * 55)
for _, row in merged.iterrows():
    print(f"\n  {row['school_id']}  [Score: {row['avg_health_score']:.0f}  Grade: {row['grade']}]")
    for rec in row["recommendation"].split(" | "):
        print(f"    → {rec}")

# ── Save full health report ───────────────────────────────────────────────────
merged.to_csv("outputs/health_report.csv", index=False)
print("\n\n💾 Saved → outputs/health_report.csv")

# ── Final system summary ──────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("SYSTEM-WIDE HEALTH SUMMARY")
print("=" * 55)
print(f"Avg health score      : {school_health['avg_health_score'].mean():.1f} / 100")
print(f"Schools graded A/B    : {(school_health['grade'].isin(['A','B'])).sum()} / {len(school_health)}")
print(f"Schools graded D/F    : {(school_health['grade'].isin(['D','F'])).sum()} / {len(school_health)}")
print(f"Avg habit rate        : {habit_df['habit_rate_pct'].mean():.1f}%")
print(f"Avg calorie gap       : {df['cal_gap'].mean():.1f} kcal/student/day")
print(f"Best school           : {school_health.iloc[0]['school_id']} (score {school_health.iloc[0]['avg_health_score']:.1f})")
print(f"Needs most support    : {school_health.iloc[-1]['school_id']} (score {school_health.iloc[-1]['avg_health_score']:.1f})")
