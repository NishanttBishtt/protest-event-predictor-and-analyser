
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

files = [
    DATA_DIR / f"gcut_daily_features_{year}.csv"
    for year in range(2014, 2020)
]

frames = []

for file in files:
    if not file.exists():
        raise FileNotFoundError(f"Missing file: {file}")

    df = pd.read_csv(file)
    frames.append(df)
    print(f"Loaded {file.name}: {len(df):,} rows")

# Combine annual datasets
gcut = pd.concat(frames, ignore_index=True)

# Standardize types
gcut["date"] = pd.to_datetime(gcut["date"], errors="raise")
gcut["country_code_alpha3"] = (
    gcut["country_code_alpha3"].astype(str).str.strip()
)

# Check for duplicate country-date observations
duplicates = gcut.duplicated(
    subset=["country_code_alpha3", "date"]
).sum()

print("\nDuplicate country-date records:", duplicates)

if duplicates > 0:
    raise ValueError(
        "Duplicate country-date records found. "
        "Inspect before continuing."
    )

# Derive additional social-media features
gcut["unrest_tweet_ratio"] = (
    gcut["unrest_tweet_count"] / gcut["tweet_count"]
)

gcut["log_tweet_count"] = (
    __import__("numpy").log1p(gcut["tweet_count"])
)

# Sort records
gcut = gcut.sort_values(
    ["country_code_alpha3", "date"]
).reset_index(drop=True)

# Save combined dataset
output = DATA_DIR / "gcut_daily_features_2014_2019.csv"
gcut.to_csv(output, index=False)

print("\n========== COMBINED G-CUT DATA ==========")
print("Shape:", gcut.shape)
print("Countries:", gcut["country_code_alpha3"].nunique())
print("Date range:", gcut["date"].min(), "to", gcut["date"].max())
print("Total tweets:", f"{gcut['tweet_count'].sum():,}")
print("Missing values:", gcut.isnull().sum().sum())
print("Output:", output)
print("\nSample:")
print(gcut.head())
