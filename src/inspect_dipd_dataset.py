
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

FILES = {
    "Event Data": {
        "path": "data/raw/dipd/Event_Data_raw.csv",
        "header": 0
    },
    "Non Event Data": {
        "path": "data/raw/dipd/Non_Event_Data_raw.csv",
        "header": None
    }
}

for name, config in FILES.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    path = config["path"]

    if not os.path.exists(path):
        print("File not found:", path)
        continue

    df = pd.read_csv(
        path,
        header=config["header"],
        names=COLUMNS if config["header"] is None else None,
        low_memory=False
    )

    print("\nShape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nLabel distribution:")
    print(df["label"].value_counts(dropna=False))

    print("\nSample tweets:")
    for tweet in df["tweet"].dropna().head(5):
        print("-", tweet)

    print("\nData types:")
    print(df.dtypes)

