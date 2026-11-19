import pandas as pd

columns = [
    "created_at", "tweet", "retweet_count",
    "follower_count", "location", "user_name",
    "statuses_count", "label"
]

event = pd.read_csv("data/raw/dipd/Event_Data_raw.csv")

non_event = pd.read_csv(
    "data/raw/dipd/Non_Event_Data_raw.csv",
    header=None,
    names=columns
)

df = pd.concat([event, non_event], ignore_index=True)

df["tweet"] = df["tweet"].fillna("").astype(str).str.strip()
df = df[df["tweet"] != ""]

label_counts = df.groupby("tweet")["label"].nunique()

conflicts = label_counts[label_counts > 1]

print("Total records:", len(df))
print("Unique tweets:", df["tweet"].nunique())
print("Duplicate occurrences:", len(df) - df["tweet"].nunique())
print("Conflicting tweets:", len(conflicts))

print("\nConflicting label combinations:")
print(
    df[df["tweet"].isin(conflicts.index)]
    .groupby("tweet")["label"]
    .agg(list)
    .value_counts()
    .head(10)
)
