
import os
import joblib
import pandas as pd


# -----------------------------------
# 1. Load trained models
# -----------------------------------

historical_model = joblib.load(
    "models/protest_model_historical.joblib"
)

combined_model = joblib.load(
    "models/protest_model_combined.joblib"
)


# -----------------------------------
# 2. Extract feature importances
# -----------------------------------

historical_importance = historical_model.feature_importances_
combined_importance = combined_model.feature_importances_

historical_features = historical_model.feature_names_in_
combined_features = combined_model.feature_names_in_


# -----------------------------------
# 3. Create importance DataFrames
# -----------------------------------

historical_df = pd.DataFrame({
    "feature": historical_features,
    "importance": historical_importance
})

historical_df = historical_df.sort_values(
    by="importance",
    ascending=False
)

combined_df = pd.DataFrame({
    "feature": combined_features,
    "importance": combined_importance
})

combined_df = combined_df.sort_values(
    by="importance",
    ascending=False
)


# -----------------------------------
# 4. Display results
# -----------------------------------

print("\n========== MODEL A: HISTORICAL ONLY ==========")

print(
    historical_df.to_string(
        index=False,
        formatters={"importance": "{:.5f}".format}
    )
)


print("\n========== MODEL B: HISTORICAL + SOCIAL MEDIA ==========")

print(
    combined_df.to_string(
        index=False,
        formatters={"importance": "{:.5f}".format}
    )
)


# -----------------------------------
# 5. Save results
# -----------------------------------

os.makedirs("results", exist_ok=True)

historical_df.to_csv(
    "results/historical_feature_importance.csv",
    index=False
)

combined_df.to_csv(
    "results/combined_feature_importance.csv",
    index=False
)

print("\nFeature importance results saved successfully.")
