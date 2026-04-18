# Mid-Day Meal System — Data Analysis & Optimisation

> End-to-end data science project covering synthetic data generation, cleaning, EDA, anomaly detection, calorie precision, health scoring, habit tracking, and meal allocation optimisation for government school mid-day meal programmes.

---

## Project structure

```
midday_meal_project/
├── data/
│   ├── midday_meal_dataset.csv        ← raw generated dataset
│   ├── midday_meal_cleaned.csv        ← cleaned dataset
│   ├── midday_meal_optimised.csv      ← with supply status & gap columns
│   └── school_supply_report.csv       ← per-school anomaly report
├── outputs/
│   └── (generated CSVs saved here)
├── 01_generate_dataset.py             ← synthetic data generation
├── 02_clean_dataset.py                ← 7-step data cleaning pipeline
├── 03_explore_dataset.py              ← basic EDA: load, inspect, stats
├── 04_eda_analysis.py                 ← deep EDA: averages, correlation, complaints
├── 05_meal_optimization.py            ← supply detection & allocation optimisation
├── 06_health_scoring.py               ← health scores, calorie precision, habit tracking
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/your-username/midday-meal-analysis.git
cd midday-meal-analysis
pip install -r requirements.txt

python 01_generate_dataset.py
python 02_clean_dataset.py
python 03_explore_dataset.py
python 04_eda_analysis.py
python 05_meal_optimization.py
python 06_health_scoring.py
```

---

## Key findings

| Metric | Value |
|---|---|
| Supply efficiency | 93.6% |
| Total meals gap | 3,093 across 250 days |
| Under-supply days | 163 / 250 (65.2%) |
| Quality–complaints correlation | r = −0.463 |
| Avg calorie gap | −12.4 meals/school/day |

---

## Tech stack

**Python 3 · Pandas · NumPy · Matplotlib · Seaborn**

Skills: data generation · cleaning pipelines · EDA · correlation analysis · threshold-based optimisation · anomaly detection · health scoring · calorie tracking
