
import pandas as pd
import joblib

DATA_PATH = "data/processed/daily_protest_forecasting.csv"
INDIA_DEMO_PATH = "data/processed/india_synthetic_demo.csv"
MODEL_PATH = "models/protest_forecasting_model.joblib"


def get_historical_forecast(country_code):
    country_code = country_code.upper()

    # Select the appropriate dataset
    if country_code == "IND":
        df = pd.read_csv(INDIA_DEMO_PATH, parse_dates=["event_date"])
        country_data = df[df["iso3"] == "IND_SYNTHETIC"]
        synthetic = True
    else:
        df = pd.read_csv(DATA_PATH, parse_dates=["event_date"])
        country_data = df[df["iso3"] == country_code]
        synthetic = False

    if country_data.empty:
        return {"error": "Country not found in forecasting dataset"}

    latest = country_data.sort_values("event_date").iloc[-1]

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]
    threshold = saved["threshold"]

    X = pd.DataFrame([latest[features].to_dict()], columns=features)
    probability = float(model.predict_proba(X)[0][1])

    result = {
        "country": country_code,
        "data_date": latest["event_date"].strftime("%Y-%m-%d"),
        "data_period": "2014-2019",
        "past_7d_events": int(latest["past_7d_events"]),
        "past_14d_events": int(latest["past_14d_events"]),
        "past_28d_events": int(latest["past_28d_events"]),
        "historical_model_probability": round(probability, 4),
        "predicted_protest": int(probability >= threshold),
        "threshold": round(float(threshold), 2),
        "is_synthetic": synthetic
    }

    if synthetic:
        result["display_name"] = "India (Synthetic Demo)"
        result["proxy_source"] = "Bangladesh (BGD)"

    return result


if __name__ == "__main__":
    print(get_historical_forecast("IND"))
