
import pandas as pd
import re
import os

df = pd.read_csv("data/processed/dipd_detection.csv")

# Select event-labelled tweets containing protest-related terms
keywords = (
    r"\bprotest\w*\b|\bdemonstrat\w*\b|\bmarch\w*\b|"
    r"\brall\w*\b|\bstrik\w*\b|\brally\b"
)

sample = df[
    (df["label"] == 1) &
    (df["tweet"].str.contains(keywords, case=False, na=False, regex=True))
].copy()

print("Matching tweets:", len(sample))

# Select a reproducible sample
sample = sample.sample(
    n=min(30, len(sample)),
    random_state=42
).reset_index(drop=True)

# Add annotation columns
sample["event_type"] = ""
sample["people"] = ""
sample["organisations"] = ""
sample["groups"] = ""
sample["locations"] = ""
sample["venues"] = ""
sample["dates"] = ""
sample["times"] = ""
sample["protest_cause"] = ""

os.makedirs("data/processed", exist_ok=True)

sample.to_csv(
    "data/processed/extraction_annotation_sample.csv",
    index=False
)

print("\nSample saved successfully.")
print("Number of tweets selected:", len(sample))
print("\nSample preview:")
print(sample[["tweet", "label"]].head(5).to_string(index=False))
