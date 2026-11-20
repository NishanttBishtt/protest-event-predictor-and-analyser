import argparse
import gzip
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


def prepare_gcut_features(input_path, output_path):
    # Each key is (country, UTC calendar date).
    # Values: total tweets, unrest-related tweets, score sum, max score.
    daily = defaultdict(lambda: [0, 0, 0.0, 0.0])
    total_records = 0
    skipped_records = 0

    print(f"Reading: {input_path}")

    # Stream the compressed JSONL file instead of loading it all into RAM.
    with gzip.open(input_path, "rt", encoding="utf-8") as file:
        for line in file:
            try:
                tweet = json.loads(line)

                country = tweet.get("country_code_alpha3")
                created_at = tweet.get("created_at")
                score = tweet.get("civil_unrest_score")
                related = tweet.get("civil_unrest_related")

                if country is None or created_at is None or score is None:
                    skipped_records += 1
                    continue

                # Convert timestamp to a UTC calendar date.
                timestamp = datetime.strptime(
                    created_at, "%a %b %d %H:%M:%S %z %Y"
                )
                date = timestamp.date().isoformat()

                score = float(score)
                key = (country, date)

                daily[key][0] += 1
                daily[key][1] += int(bool(related))
                daily[key][2] += score
                daily[key][3] = max(daily[key][3], score)

                total_records += 1

                if total_records % 1_000_000 == 0:
                    print(f"Processed {total_records:,} records")

            except (json.JSONDecodeError, ValueError, TypeError):
                skipped_records += 1

    rows = []

    for (country, date), values in daily.items():
        tweet_count, unrest_count, score_sum, max_score = values

        rows.append({
            "country_code_alpha3": country,
            "date": date,
            "tweet_count": tweet_count,
            "unrest_tweet_count": unrest_count,
            "mean_unrest_score": score_sum / tweet_count,
            "max_unrest_score": max_score,
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            ["country_code_alpha3", "date"]
        ).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\nAggregation complete.")
    print(f"Valid records processed: {total_records:,}")
    print(f"Skipped records: {skipped_records:,}")
    print(f"Daily country-date rows: {len(df):,}")
    print(f"Countries: {df['country_code_alpha3'].nunique() if not df.empty else 0}")
    print(f"Output saved to: {output_path}")

    if not df.empty:
        print("\nSample rows:")
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prepare_gcut_features(args.input, args.output)
