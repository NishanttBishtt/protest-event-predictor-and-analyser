
import pandas as pd
import os
import sys
import re

sys.path.insert(0, "src")
from event_information_extraction import extract_event_information

df = pd.read_csv(
    "data/processed/extraction_annotation_sample.csv"
).fillna("")

# Evaluate only tweets describing protest-related events
df = df[df["is_event_description"] == 1].copy()

fields = [
    "event_type",
    "people",
    "organisations",
    "groups",
    "locations",
    "venues",
    "dates",
    "times",
    "protest_cause"
]

def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())

def split_values(value):
    if not str(value).strip():
        return set()
    return {
        normalize(v)
        for v in str(value).split(";")
        if normalize(v)
    }

scores = {
    field: {"tp": 0, "fp": 0, "fn": 0}
    for field in fields
}

results = []

for _, row in df.iterrows():
    prediction = extract_event_information(row["tweet"])

    result = {"tweet": row["tweet"]}

    for field in fields:
        actual = split_values(row[field])
        predicted = prediction.get(field, [])

        if isinstance(predicted, list):
            predicted = {normalize(v) for v in predicted}
        elif predicted:
            predicted = {normalize(predicted)}
        else:
            predicted = set()

        tp = len(actual & predicted)
        fp = len(predicted - actual)
        fn = len(actual - predicted)

        scores[field]["tp"] += tp
        scores[field]["fp"] += fp
        scores[field]["fn"] += fn

        result[f"{field}_actual"] = "; ".join(sorted(actual))
        result[f"{field}_predicted"] = "; ".join(sorted(predicted))

    results.append(result)

print("\nFIELD-LEVEL EXTRACTION EVALUATION")
print("=" * 70)

for field, score in scores.items():
    tp = score["tp"]
    fp = score["fp"]
    fn = score["fn"]

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print(
        f"{field:18} "
        f"P={precision:.3f} "
        f"R={recall:.3f} "
        f"F1={f1:.3f} "
        f"TP={tp} FP={fp} FN={fn}"
    )

os.makedirs("results", exist_ok=True)

pd.DataFrame(results).to_csv(
    "results/event_extraction_evaluation.csv",
    index=False
)

print("\nDetailed predictions saved.")
