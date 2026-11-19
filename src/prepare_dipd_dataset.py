
import pandas as pd
import os

COLUMNS = [
    "created_at",
    "tweet",
    "retweet_count",
    "follower_count",
    "location",
    "user_name",
    "statuses_count",
    "label"
]

event_df = pd.read_csv(
    "data/raw/dipd/Event_Data_raw.csv",
    low_memory=False
)

non_event_df = pd.read_csv(
    "data/raw/dipd/Non_Event_Data_raw.csv",
    header=None,
    names=COLUMNS,
    low_memory=False
)

df = pd.concat([event_df, non_event_df], ignore_index=True)

# Clean tweet text
df["tweet"] = df["tweet"].fillna("").astype(str).str.strip()
df = df[df["tweet"] != ""]

# Remove tweets having conflicting labels
label_counts = df.groupby("tweet")["label"].nunique()
conflicting_tweets = label_counts[label_counts > 1].index

print("Conflicting tweets removed:", len(conflicting_tweets))

df = df[~df["tweet"].isin(conflicting_tweets)]

# Remove remaining exact duplicates
df = df.drop_duplicates(subset="tweet")

# Retain classification columns
df = df[["tweet", "label"]]

print("\nFinal dataset shape:", df.shape)
print("\nFinal label distribution:")
print(df["label"].value_counts())

# Save
os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/dipd_detection.csv",
    index=False
)

print("\nDataset saved successfully.")
