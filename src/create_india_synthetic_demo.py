
import pandas as pd

SOURCE_PATH = "data/processed/daily_protest_forecasting.csv"
OUTPUT_PATH = "data/processed/india_synthetic_demo.csv"

# Load original forecasting data
df = pd.read_csv(SOURCE_PATH)

# Select Bangladesh records
bangladesh = df[df["iso3"] == "BGD"].copy()

if bangladesh.empty:
    raise ValueError("Bangladesh records not found.")

# Create synthetic India records
india = bangladesh.copy()
india["iso3"] = "IND_SYNTHETIC"

# Add provenance information
india["source_country"] = "BGD"
india["is_synthetic"] = True

# Save separately
india.to_csv(OUTPUT_PATH, index=False)

print("Synthetic India dataset created.")
print("Source country: Bangladesh (BGD)")
print("Synthetic records:", len(india))
print("Date range:", india["event_date"].min(), "to", india["event_date"].max())
print("Output:", OUTPUT_PATH)
