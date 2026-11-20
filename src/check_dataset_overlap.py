
import pandas as pd

ACLED_PATH = "data/processed/daily_protest_forecasting.csv"
GCUT_PATH = "data/processed/gcut_daily_features_2014_2019.csv"

acled = pd.read_csv(ACLED_PATH)
gcut = pd.read_csv(GCUT_PATH)

# Standardize join columns
acled["event_date"] = pd.to_datetime(acled["event_date"])
gcut["date"] = pd.to_datetime(gcut["date"])

acled = acled.rename(columns={
    "iso3": "country",
    "event_date": "date"
})

gcut = gcut.rename(columns={
    "country_code_alpha3": "country"
})

# Country-level overlap
acled_countries = set(acled["country"].unique())
gcut_countries = set(gcut["country"].unique())

print("========== COUNTRY OVERLAP ==========")
print("ACLED countries:", len(acled_countries))
print("G-CUT countries:", len(gcut_countries))
print("Common countries:", len(acled_countries & gcut_countries))
print("ACLED-only countries:", sorted(acled_countries - gcut_countries))
print("G-CUT-only countries:", sorted(gcut_countries - acled_countries))

# Check duplicate keys before joining
print("\n========== DUPLICATE CHECK ==========")
print("ACLED duplicates:",
      acled.duplicated(["country", "date"]).sum())
print("G-CUT duplicates:",
      gcut.duplicated(["country", "date"]).sum())

# Left join: preserve all ACLED observations
merged = acled.merge(
    gcut,
    on=["country", "date"],
    how="left",
    validate="one_to_one",
    indicator=True
)

print("\n========== DATE OVERLAP ==========")
print("ACLED rows:", len(acled))
print("Matched rows:", (merged["_merge"] == "both").sum())
print("Unmatched ACLED rows:", (merged["_merge"] == "left_only").sum())
print("Match percentage:",
      round((merged["_merge"] == "both").mean() * 100, 2), "%")

# Coverage by country
coverage = (
    merged.groupby("country")
    .agg(
        acled_days=("date", "size"),
        matched_days=("_merge", lambda x: (x == "both").sum())
    )
)

coverage["match_percentage"] = (
    coverage["matched_days"] / coverage["acled_days"] * 100
)

print("\n========== COVERAGE BY COUNTRY ==========")
print(coverage.sort_values("match_percentage").to_string())

# Save diagnostic results
coverage.to_csv("results/gcut_acled_country_coverage.csv")

print("\nCoverage report saved to:")
print("results/gcut_acled_country_coverage.csv")
