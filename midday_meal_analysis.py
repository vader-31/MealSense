"""
midday_meal_analysis.py
=======================
Mid-Day Meal System — Complete Data Analysis & Optimisation Pipeline
=====================================================================
Run this single file to execute the entire project end-to-end:

  Step 1  → Generate synthetic dataset (250 rows, 20 schools)
  Step 2  → Clean data (7-step pipeline)
  Step 3  → Explore dataset (shape, missing values, stats)
  Step 4  → EDA (school averages, complaints, correlation + charts)
  Step 5  → Meal optimisation (supply detection, allocation + charts)
  Step 6  → Health scoring, calorie precision, habit tracking,
              personalised recommendations + charts

Output files
────────────
  data/midday_meal_dataset.csv
  data/midday_meal_cleaned.csv
  data/midday_meal_optimised.csv
  data/school_supply_report.csv
  outputs/eda_school_averages.png
  outputs/eda_complaints_distribution.png
  outputs/eda_correlation_heatmap.png
  outputs/opt_supply_status.png
  outputs/opt_school_deficit.png
  outputs/calorie_precision.png
  outputs/health_scores.png
  outputs/habit_heatmap.png
  outputs/health_report.csv

Requirements
────────────
  pip install pandas numpy matplotlib seaborn

Usage
─────
  python midday_meal_analysis.py
"""

import os
import random
from datetime import date, timedelta

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Global config ─────────────────────────────────────────────────────────────
os.makedirs("data",    exist_ok=True)
os.makedirs("outputs", exist_ok=True)

np.random.seed(42)
random.seed(42)

DIVIDER = "=" * 60


def section(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — GENERATE SYNTHETIC DATASET
# ══════════════════════════════════════════════════════════════════════════════

def generate_dataset():
    section("STEP 1 — GENERATE SYNTHETIC DATASET")

    N_ROWS     = 250
    N_SCHOOLS  = 20
    START_DATE = date(2024, 6, 1)
    DATE_RANGE = 180

    school_ids = [f"SCH{str(i).zfill(3)}" for i in range(1, N_SCHOOLS + 1)]

    # Each school has its own typical attendance band
    school_profiles = {
        sid: {
            "avg_att": random.randint(120, 280),
            "std_att": random.randint(15, 45),
        }
        for sid in school_ids
    }

    # Build weekday date pool
    all_dates = [START_DATE + timedelta(days=i) for i in range(DATE_RANGE)]
    weekdays  = [d for d in all_dates if d.weekday() < 5]

    rows = []
    for i in range(N_ROWS):
        school_id = random.choice(school_ids)
        profile   = school_profiles[school_id]

        attendance = int(np.clip(
            np.random.normal(profile["avg_att"], profile["std_att"]),
            40, 320
        ))

        # Meals served: slightly below attendance (kitchen under-prepares ~3%)
        meals_served = max(0, attendance + random.randint(-12, 0))

        # Quality: triangular distribution skewed toward 7
        food_quality_score = round(random.triangular(3.0, 10.0, 7.0), 1)

        # Complaints: inversely correlated with quality
        base = max(0, int((10 - food_quality_score) * random.uniform(0, 1.8)))
        complaints_count = max(0, base + random.randint(-1, 2))

        rows.append({
            "student_id":         f"STU{str(i + 1).zfill(4)}",
            "school_id":          school_id,
            "date":               random.choice(weekdays).strftime("%Y-%m-%d"),
            "attendance":         attendance,
            "meals_served":       meals_served,
            "food_quality_score": food_quality_score,
            "complaints_count":   complaints_count,
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.to_csv("data/midday_meal_dataset.csv", index=False)

    print(f"Rows generated   : {len(df)}")
    print(f"Schools          : {df['school_id'].nunique()}")
    print(f"Date range       : {df['date'].min()}  →  {df['date'].max()}")
    print(f"Avg attendance   : {df['attendance'].mean():.1f}")
    print(f"Avg meals served : {df['meals_served'].mean():.1f}")
    print("Saved → data/midday_meal_dataset.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CLEAN DATASET (7-step pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def clean_dataset():
    section("STEP 2 — CLEAN DATASET")

    df = pd.read_csv("data/midday_meal_dataset.csv")
    print(f"Rows loaded: {len(df)}")

    # 1. Fix data types
    df["date"]               = pd.to_datetime(df["date"])
    df["attendance"]         = df["attendance"].astype(int)
    df["meals_served"]       = df["meals_served"].astype(int)
    df["food_quality_score"] = df["food_quality_score"].astype(float)
    df["complaints_count"]   = df["complaints_count"].astype(int)
    df["student_id"]         = df["student_id"].astype(str).str.strip()
    df["school_id"]          = df["school_id"].astype(str).str.strip()
    print("✅  Step 1 — data types fixed")

    # 2. Handle missing values
    missing_before = df.isnull().sum().sum()
    df["attendance"]         = df["attendance"].fillna(df["attendance"].median())
    df["meals_served"]       = df["meals_served"].fillna(df["meals_served"].median())
    df["food_quality_score"] = df["food_quality_score"].fillna(df["food_quality_score"].median())
    df["complaints_count"]   = df["complaints_count"].fillna(0)
    print(f"✅  Step 2 — missing values: {missing_before} → {df.isnull().sum().sum()}")

    # 3. Remove duplicates
    dupes = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"✅  Step 3 — duplicate rows removed: {dupes}")

    # 4. Fix logical inconsistency: meals_served > attendance
    inconsistent = (df["meals_served"] > df["attendance"]).sum()
    df["meals_served"] = df[["meals_served", "attendance"]].min(axis=1)
    print(f"✅  Step 4 — meals_served > attendance fixed: {inconsistent} rows")

    # 5. Clamp values to valid ranges
    df["food_quality_score"] = df["food_quality_score"].clip(1, 10)
    df["attendance"]         = df["attendance"].clip(lower=0)
    df["complaints_count"]   = df["complaints_count"].clip(lower=0)
    print("✅  Step 5 — values clamped to valid ranges")

    # 6. Standardise text columns
    df["student_id"] = df["student_id"].str.strip().str.upper()
    df["school_id"]  = df["school_id"].str.strip().str.upper()
    print("✅  Step 6 — text columns standardised")

    # 7. Sort and reset index
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("✅  Step 7 — sorted by date, index reset")

    print(f"\nFinal rows       : {len(df)}")
    print(f"Missing values   : {df.isnull().sum().sum()}")
    print(f"Schools covered  : {df['school_id'].nunique()}")
    df.to_csv("data/midday_meal_cleaned.csv", index=False)
    print("Saved → data/midday_meal_cleaned.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — EXPLORE DATASET
# ══════════════════════════════════════════════════════════════════════════════

def explore_dataset():
    section("STEP 3 — EXPLORE DATASET")

    df = pd.read_csv("data/midday_meal_cleaned.csv")

    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    print("First 5 rows:")
    print(df.head().to_string())

    print("\nMissing values per column:")
    print(df.isnull().sum())
    status = "✅  No missing values." if df.isnull().sum().sum() == 0 else f"⚠️   Total missing: {df.isnull().sum().sum()}"
    print(status)

    print("\nBasic statistics:")
    print(df.describe().round(2))

    print(f"\nUnique students : {df['student_id'].nunique()}")
    print(f"Unique schools  : {df['school_id'].nunique()}")
    print(f"Date range      : {df['date'].min()}  →  {df['date'].max()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — EDA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def eda_analysis():
    section("STEP 4 — EDA ANALYSIS")

    df = pd.read_csv("data/midday_meal_cleaned.csv")

    # ── Average meals per school ──────────────────────────────────────────────
    avg_meals = (
        df.groupby("school_id")["meals_served"]
        .mean().round(2).sort_values(ascending=False).reset_index()
    )
    avg_meals.columns = ["school_id", "avg_meals_served"]
    print("Average meals served per school:")
    print(avg_meals.to_string(index=False))

    # ── Average attendance per school ─────────────────────────────────────────
    avg_att = (
        df.groupby("school_id")["attendance"]
        .mean().round(2).sort_values(ascending=False).reset_index()
    )
    avg_att.columns = ["school_id", "avg_attendance"]
    print("\nAverage attendance per school:")
    print(avg_att.to_string(index=False))

    # ── Chart: school averages ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    att_aligned = avg_att.set_index("school_id").loc[avg_meals["school_id"], "avg_attendance"].values
    ax.barh(avg_meals["school_id"], avg_meals["avg_meals_served"],
            color="#378ADD", alpha=0.85, label="Avg meals served")
    ax.barh(avg_meals["school_id"], att_aligned,
            color="#D3D1C7", alpha=0.55, label="Avg attendance")
    ax.set_xlabel("Count")
    ax.set_title("Average meals served vs attendance per school")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/eda_school_averages.png", dpi=150)
    plt.close()
    print("\n📊 Saved → outputs/eda_school_averages.png")

    # ── Complaints distribution ───────────────────────────────────────────────
    print("\nComplaints distribution:")
    print(df["complaints_count"].describe().round(2))
    print("\nDays per complaint level:")
    print(df["complaints_count"].value_counts().sort_index().to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    vals   = df["complaints_count"].value_counts().sort_index()
    colors = ["#27500A" if v <= 3 else "#BA7517" if v <= 6 else "#A32D2D"
              for v in vals.index]
    ax.bar(vals.index, vals.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Complaints on a given day")
    ax.set_ylabel("Number of days")
    ax.set_title("Complaints distribution across all school-days")
    ax.set_xticks(vals.index)
    plt.tight_layout()
    plt.savefig("outputs/eda_complaints_distribution.png", dpi=150)
    plt.close()
    print("📊 Saved → outputs/eda_complaints_distribution.png")

    # ── Correlation matrix ────────────────────────────────────────────────────
    cols = ["attendance", "meals_served", "food_quality_score", "complaints_count"]
    corr = df[cols].corr().round(3)
    print("\nCorrelation matrix:")
    print(corr.to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation matrix — numeric columns")
    plt.tight_layout()
    plt.savefig("outputs/eda_correlation_heatmap.png", dpi=150)
    plt.close()
    print("📊 Saved → outputs/eda_correlation_heatmap.png")

    # ── Key insights ──────────────────────────────────────────────────────────
    print("\nKey correlation insights:")
    pairs = [
        (cols[i], cols[j], corr.iloc[i, j])
        for i in range(len(cols))
        for j in range(i + 1, len(cols))
    ]
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for c1, c2, val in pairs:
        strength  = "strong" if abs(val) >= 0.4 else "moderate" if abs(val) >= 0.2 else "weak / no"
        direction = "positive" if val > 0 else "negative"
        print(f"  {c1} vs {c2}: {val:+.3f}  →  {strength} {direction}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — MEAL OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════

def meal_optimization():
    section("STEP 5 — MEAL OPTIMISATION")

    df     = pd.read_csv("data/midday_meal_cleaned.csv")
    BUFFER = 0.05

    # ── Estimate required & recommended meals ─────────────────────────────────
    df["required_meals"]    = df["attendance"]
    df["recommended_meals"] = (df["attendance"] * (1 + BUFFER)).astype(int)

    print(f"Buffer               : {int(BUFFER * 100)}%")
    print(f"Avg attendance       : {df['attendance'].mean():.1f}")
    print(f"Avg required meals   : {df['required_meals'].mean():.1f}")
    print(f"Avg recommended meals: {df['recommended_meals'].mean():.1f}")
    print(f"Avg meals served     : {df['meals_served'].mean():.1f}")

    # ── Classify supply status ────────────────────────────────────────────────
    def classify(row):
        if row["meals_served"] > row["recommended_meals"]: return "over-supply"
        if row["meals_served"] < row["required_meals"]:    return "under-supply"
        return "optimal"

    df["supply_status"] = df.apply(classify, axis=1)
    df["gap"]           = df["meals_served"] - df["recommended_meals"]

    sc = df["supply_status"].value_counts()
    print(f"\nOptimal days      : {sc.get('optimal', 0)}")
    print(f"Under-supply days : {sc.get('under-supply', 0)}")
    print(f"Over-supply days  : {sc.get('over-supply', 0)}")
    print(f"Total meals short : {int(df.loc[df['gap']<0,'gap'].abs().sum())}")
    print(f"Total excess meals: {int(df.loc[df['gap']>0,'gap'].sum())}")
    print(f"Avg daily gap     : {df['gap'].mean():.1f} meals")

    # ── Pie chart ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = {"optimal": "#639922", "under-supply": "#E24B4A", "over-supply": "#BA7517"}
    sc2  = sc.reindex(["optimal", "under-supply", "over-supply"]).dropna()
    ax.pie(sc2.values, labels=sc2.index, autopct="%1.1f%%",
           colors=[cmap[k] for k in sc2.index], startangle=90)
    ax.set_title("Supply status across all school-days")
    plt.tight_layout()
    plt.savefig("outputs/opt_supply_status.png", dpi=150)
    plt.close()
    print("📊 Saved → outputs/opt_supply_status.png")

    # ── Per-school anomaly report ─────────────────────────────────────────────
    df["daily_deficit"] = (df["recommended_meals"] - df["meals_served"]).clip(lower=0)
    df["daily_excess"]  = (df["meals_served"] - df["recommended_meals"]).clip(lower=0)

    school_report = (
        df.groupby("school_id").agg(
            avg_attendance    = ("attendance",        "mean"),
            avg_meals_served  = ("meals_served",      "mean"),
            avg_recommended   = ("recommended_meals", "mean"),
            avg_daily_deficit = ("daily_deficit",     "mean"),
            avg_daily_excess  = ("daily_excess",      "mean"),
            under_supply_days = ("supply_status",     lambda x: (x == "under-supply").sum()),
            over_supply_days  = ("supply_status",     lambda x: (x == "over-supply").sum()),
            optimal_days      = ("supply_status",     lambda x: (x == "optimal").sum()),
        ).round(2).reset_index()
    )
    print("\nPer-school anomaly report:")
    print(school_report.to_string(index=False))

    # ── Suggested allocation ──────────────────────────────────────────────────
    alloc = school_report[["school_id", "avg_attendance", "avg_meals_served",
                            "avg_recommended", "avg_daily_deficit"]].copy()
    alloc["suggested_daily_allocation"] = (alloc["avg_attendance"] * (1 + BUFFER)).round().astype(int)
    alloc["meals_to_add_per_day"]       = (alloc["suggested_daily_allocation"] - alloc["avg_meals_served"]).round(1)
    alloc = alloc.sort_values("avg_daily_deficit", ascending=False).reset_index(drop=True)
    alloc.index += 1
    print("\nSuggested daily allocation (ranked by urgency):")
    print(alloc.to_string())

    # ── Deficit bar chart ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(alloc["school_id"], alloc["avg_daily_deficit"],
           color="#E24B4A", alpha=0.85)
    ax.set_xlabel("School")
    ax.set_ylabel("Avg daily deficit (meals)")
    ax.set_title("Average daily meal deficit per school")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/opt_school_deficit.png", dpi=150)
    plt.close()
    print("📊 Saved → outputs/opt_school_deficit.png")

    # ── Top 10 worst days ─────────────────────────────────────────────────────
    worst = (
        df[df["supply_status"] == "under-supply"]
        [["date", "school_id", "attendance", "meals_served",
          "recommended_meals", "gap", "food_quality_score", "complaints_count"]]
        .sort_values("gap").head(10).reset_index(drop=True)
    )
    worst.index += 1
    print("\nTop 10 worst under-supply days:")
    print(worst.to_string())

    # ── Efficiency summary ────────────────────────────────────────────────────
    curr = df["meals_served"].sum()
    rec  = df["recommended_meals"].sum()
    print(f"\nTotal meals served     : {curr:,}")
    print(f"Total recommended      : {rec:,}")
    print(f"Supply efficiency      : {curr/rec*100:.1f}%")
    print(f"Meals gap to close     : {rec-curr:,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv("data/midday_meal_optimised.csv", index=False)
    school_report.to_csv("data/school_supply_report.csv", index=False)
    print("Saved → data/midday_meal_optimised.csv")
    print("Saved → data/school_supply_report.csv")
    return df, school_report


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — HEALTH SCORING, CALORIE PRECISION, HABIT TRACKING, PERSONALISATION
# ══════════════════════════════════════════════════════════════════════════════

def health_scoring():
    section("STEP 6 — HEALTH SCORING & ADVANCED ANALYTICS")

    df     = pd.read_csv("data/midday_meal_optimised.csv")
    report = pd.read_csv("data/school_supply_report.csv")

    CAL_TARGET = 650   # kcal per student per day (ICDS primary standard)

    # ── Calorie precision ─────────────────────────────────────────────────────
    # Proxy: cal_per_meal = 600 + quality_score × 8
    # (higher quality → richer ingredients → more calories)
    df["cal_per_meal"]    = 600 + df["food_quality_score"] * 8
    df["cal_per_student"] = np.where(
        df["meals_served"] > 0,
        (df["cal_per_meal"]).round(1),
        0
    )
    df["cal_target"] = CAL_TARGET
    df["cal_gap"]    = (df["cal_per_student"] - df["cal_target"]).round(1)
    df["cal_precision_pct"] = (df["cal_per_student"] / df["cal_target"] * 100).round(1)

    cal_summary = (
        df.groupby("school_id").agg(
            avg_cal_per_student = ("cal_per_student",   "mean"),
            avg_cal_gap         = ("cal_gap",            "mean"),
            avg_precision_pct   = ("cal_precision_pct", "mean"),
            days_on_target      = ("cal_gap",            lambda x: (x.abs() <= 30).sum()),
            total_days          = ("cal_gap",            "count"),
        ).round(2).reset_index()
    )
    cal_summary["on_target_pct"] = (
        cal_summary["days_on_target"] / cal_summary["total_days"] * 100
    ).round(1)

    print("Calorie precision (system):")
    print(f"  Target          : {CAL_TARGET} kcal/student/day")
    print(f"  Avg actual      : {df['cal_per_student'].mean():.1f} kcal")
    print(f"  Avg gap         : {df['cal_gap'].mean():.1f} kcal")
    print(f"  Days within ±30 : {(df['cal_gap'].abs()<=30).sum()} / {len(df)}")
    print("\nPer-school calorie summary:")
    print(cal_summary.to_string(index=False))

    # ── Calorie precision chart ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = cal_summary["avg_cal_gap"].apply(
        lambda g: "#639922" if abs(g) <= 30 else "#BA7517" if abs(g) <= 60 else "#E24B4A"
    )
    ax.bar(cal_summary["school_id"], cal_summary["avg_cal_per_student"],
           color=colors.values, edgecolor="white", linewidth=0.5)
    ax.axhline(CAL_TARGET, color="#185FA5", linestyle="--",
               linewidth=1.5, label=f"Target {CAL_TARGET} kcal")
    ax.set_xlabel("School")
    ax.set_ylabel("Avg calories per student")
    ax.set_title("Calorie precision — avg calories per student vs ICDS target")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/calorie_precision.png", dpi=150)
    plt.close()
    print("📊 Saved → outputs/calorie_precision.png")

    # ── Health scoring ────────────────────────────────────────────────────────
    # Composite score (0–100) from 5 weighted dimensions:
    #   food quality       30%
    #   calorie precision  25%
    #   supply efficiency  20%
    #   complaint load     15%
    #   service consistency 10%
    W = dict(quality=0.30, calories=0.25, efficiency=0.20,
             complaints=0.15, consistency=0.10)

    def health_score_row(row):
        q  = (row["food_quality_score"] - 1) / 9 * 100
        c  = max(0, 100 - abs(row["cal_gap"]) / row["cal_target"] * 200)
        e  = min(100, row["meals_served"] / max(row["recommended_meals"], 1) * 100)
        cp = max(0, 100 - row["complaints_count"] * 10)
        cs = 100 if row["meals_served"] > 0 else 0
        return round(
            W["quality"] * q + W["calories"] * c + W["efficiency"] * e +
            W["complaints"] * cp + W["consistency"] * cs, 1
        )

    df["health_score"] = df.apply(health_score_row, axis=1)

    school_health = (
        df.groupby("school_id").agg(
            avg_health_score = ("health_score",       "mean"),
            avg_quality      = ("food_quality_score",  "mean"),
            avg_efficiency   = ("cal_precision_pct",  "mean"),
            avg_complaints   = ("complaints_count",   "mean"),
        ).round(2).reset_index().sort_values("avg_health_score", ascending=False)
    )

    def grade(s):
        if s >= 80: return "A"
        if s >= 70: return "B"
        if s >= 60: return "C"
        if s >= 50: return "D"
        return "F"

    school_health["grade"] = school_health["avg_health_score"].apply(grade)

    print("\nHealth scores per school:")
    print(school_health.to_string(index=False))

    # ── Health score chart ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    palette = school_health["avg_health_score"].apply(
        lambda s: "#639922" if s >= 75 else "#BA7517" if s >= 55 else "#E24B4A"
    )
    bars = ax.bar(school_health["school_id"], school_health["avg_health_score"],
                  color=palette.values, edgecolor="white")
    ax.axhline(75, color="#185FA5", linestyle="--", linewidth=1, label="Good (75)")
    ax.axhline(55, color="#BA7517", linestyle=":",  linewidth=1, label="Warning (55)")
    for bar, g in zip(bars, school_health["grade"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                g, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlabel("School")
    ax.set_ylabel("Health score (0–100)")
    ax.set_title("Composite health score per school (A–F graded)")
    ax.set_ylim(0, 108)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/health_scores.png", dpi=150)
    plt.close()
    print("📊 Saved → outputs/health_scores.png")

    # ── Habit tracking ────────────────────────────────────────────────────────
    # Good habit day = meals_served >= required_meals AND quality >= 6
    def streaks(series):
        vals = list(series)
        cur = 0
        for v in reversed(vals):
            if v: cur += 1
            else: break
        longest = run = 0
        for v in vals:
            run = run + 1 if v else 0
            longest = max(longest, run)
        return cur, longest

    habit_records = []
    for school, grp in df.groupby("school_id"):
        grp  = grp.sort_values("date").tail(28)
        good = (grp["meals_served"] >= grp["required_meals"]) & (grp["food_quality_score"] >= 6)
        cur, mx = streaks(good)
        habit_records.append({
            "school_id":      school,
            "days_tracked":   len(grp),
            "good_days":      int(good.sum()),
            "habit_rate_pct": round(good.mean() * 100, 1),
            "current_streak": cur,
            "longest_streak": mx,
        })

    habit_df = (
        pd.DataFrame(habit_records)
        .sort_values("habit_rate_pct", ascending=False)
    )
    print("\nHabit tracking (last 28 days per school):")
    print(habit_df.to_string(index=False))

    # ── Habit heatmap ─────────────────────────────────────────────────────────
    pivot = df.copy()
    pivot["date_ord"] = pd.to_datetime(pivot["date"]).dt.strftime("%Y-%m-%d")
    pivot["good_day"] = (
        (pivot["meals_served"] >= pivot["required_meals"]) &
        (pivot["food_quality_score"] >= 6)
    ).astype(int)

    top10  = school_health["school_id"].head(10).tolist()
    hmap   = (
        pivot[pivot["school_id"].isin(top10)]
        .pivot_table(index="school_id", columns="date_ord",
                     values="good_day", aggfunc="max")
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap    = mcolors.ListedColormap(["#FEE2E2", "#D1FAE5"])
    ax.imshow(hmap.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(range(len(hmap.index)))
    ax.set_yticklabels(hmap.index, fontsize=9)
    n    = len(hmap.columns)
    step = max(1, n // 8)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels([hmap.columns[i] for i in range(0, n, step)],
                       rotation=45, fontsize=8)
    ax.set_title("Habit heatmap — green = good day (top 10 schools by health score)")
    plt.tight_layout()
    plt.savefig("outputs/habit_heatmap.png", dpi=150)
    plt.close()
    print("📊 Saved → outputs/habit_heatmap.png")

    # ── Personalised recommendations ──────────────────────────────────────────
    # Rule-based engine: actions ranked by each school's worst dimension.
    def recs(row):
        actions = []
        if row["avg_health_score"] < 55:
            actions.append("URGENT: Request district nutrition officer visit — score below 55")
        if row["avg_quality"] < 6.0:
            actions.append(
                f"Improve food quality ({row['avg_quality']:.1f}/10): "
                "add protein dish (rajma/egg) at least 3×/week"
            )
        if row["avg_complaints"] > 4:
            actions.append(
                f"Complaints high ({row['avg_complaints']:.1f}/day): "
                "review menu variety and ingredient freshness"
            )
        if row["avg_health_score"] >= 80:
            actions.append(
                "Benchmark school: document menu rotation and share as district template"
            )
        if not actions:
            actions.append(
                "Maintain current performance; pilot weekly student feedback form"
            )
        return actions

    merged = (
        school_health
        .merge(cal_summary[["school_id", "avg_cal_gap", "on_target_pct"]], on="school_id", how="left")
        .merge(habit_df[["school_id", "habit_rate_pct", "current_streak"]],  on="school_id", how="left")
    )
    merged["recommendation"] = merged.apply(lambda r: " | ".join(recs(r)), axis=1)

    print("\nPersonalised recommendations per school:")
    for _, row in merged.iterrows():
        print(f"\n  {row['school_id']}  [Score: {row['avg_health_score']:.0f}  Grade: {row['grade']}]")
        for rec in row["recommendation"].split(" | "):
            print(f"    → {rec}")

    # ── System-wide summary ───────────────────────────────────────────────────
    print(f"\nSystem-wide health summary:")
    print(f"  Avg health score   : {school_health['avg_health_score'].mean():.1f} / 100")
    print(f"  Schools graded A/B : {school_health['grade'].isin(['A','B']).sum()} / {len(school_health)}")
    print(f"  Schools graded D/F : {school_health['grade'].isin(['D','F']).sum()} / {len(school_health)}")
    print(f"  Avg habit rate     : {habit_df['habit_rate_pct'].mean():.1f}%")
    print(f"  Avg calorie gap    : {df['cal_gap'].mean():.1f} kcal/student/day")
    print(f"  Best school        : {school_health.iloc[0]['school_id']}  (score {school_health.iloc[0]['avg_health_score']:.1f})")
    print(f"  Needs most support : {school_health.iloc[-1]['school_id']}  (score {school_health.iloc[-1]['avg_health_score']:.1f})")

    merged.to_csv("outputs/health_report.csv", index=False)
    print("Saved → outputs/health_report.csv")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — run the full pipeline in sequence
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + DIVIDER)
    print("  MID-DAY MEAL SYSTEM — FULL ANALYSIS PIPELINE")
    print(DIVIDER)

    generate_dataset()
    clean_dataset()
    explore_dataset()
    eda_analysis()
    meal_optimization()
    health_scoring()

    print("\n" + DIVIDER)
    print("  PIPELINE COMPLETE")
    print(DIVIDER)
    print("\nOutputs saved:")
    print("  data/   → 4 CSV files")
    print("  outputs/ → 8 PNG charts + health_report.csv")
