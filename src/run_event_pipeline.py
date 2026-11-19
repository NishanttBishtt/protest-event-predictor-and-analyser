
import joblib
import json

from event_information_extraction import extract_event_information
from country_resolver import resolve_country
from get_historical_forecast import get_historical_forecast


# Load trained event detection model
model = joblib.load("models/event_detection_tfidf.joblib")


def process_tweet(tweet):

    # 1. EVENT DETECTION
    prediction = int(model.predict([tweet])[0])
    probabilities = model.predict_proba([tweet])[0]

    event_probability = float(probabilities[1])

    result = {
        "tweet": tweet,
        "is_event": prediction == 1,
        "event_probability": round(event_probability, 4),
        "event_information": None,
        "country_resolution": None,
        "forecast": None
    }

    # Borderline prediction review
    if 0.20 <= event_probability < 0.50:
        result["review_required"] = True
        result["message"] = (
            "Borderline prediction: manual review recommended"
        )

    # Stop if not classified as an event
    if prediction != 1:
        return result

    # 2. EVENT INFORMATION EXTRACTION
    event_info = extract_event_information(tweet)

    result["event_information"] = event_info


    # 3. COUNTRY RESOLUTION
    locations = event_info.get("locations", [])

    country_result = resolve_country(locations)

    result["country_resolution"] = country_result

    if country_result["status"] != "resolved":
        result["forecast_status"] = (
            "Forecast skipped: country could not be resolved"
        )
        return result

    country_code = country_result["country_code"]

    # 4. PROTEST FORECASTING
    forecast = get_historical_forecast(country_code)

    result["forecast"] = forecast

    if forecast.get("is_synthetic"):
        result["forecast_warning"] = (
            "Synthetic demonstration only. "
            "Bangladesh-derived data is being used "
            "as a proxy for India."
        )

    elif "error" in forecast:
        result["forecast_status"] = forecast["error"]

    else:
        result["forecast_status"] = "Historical forecast generated"

    return result




# TESTING

# if __name__ == "__main__":

#     tweet = (
#         "If Gyanesh Kumar does not resign by tomorrow, "
#         "CJP will begin its nationwide protest from Mumbai "
#         "on 2 October, and take it to cities across India. "
#         "This Gandhi Jayanti let's pledge to save democracy."
#     )

#     output = process_tweet(tweet)

#     print("\n========== COMPLETE PIPELINE ==========\n")

#     print(json.dumps(
#         output,
#         indent=4,
#         ensure_ascii=False
#     ))
