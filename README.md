# Social-Media-Based Civil Unrest Intelligence System

An integrated prototype that analyzes social media posts to detect protest-related events, extract structured event information, identify the associated country, and provide a historical protest forecast.

## Pipeline

Tweet/Post
    ↓
Event Detection
    ↓
Event Information Extraction
    ↓
Country Resolution
    ↓
Historical Protest Forecast
    ↓
Streamlit Interface

## Features

- **Event Detection:** TF-IDF feature extraction with a Logistic Regression classifier.
- **Information Extraction:** spaCy Named Entity Recognition and rule-based extraction.
- **Extracted Information:** Event type, people, organisations, groups, locations, venues, dates, times, and protest cause.
- **Country Resolution:** Location matching using a GeoNames SQLite database.
- **Protest Forecasting:** Random Forest model using historical protest activity.
- **Streamlit Interface:** Interactive interface to process tweets and view results.
- **Synthetic Demonstration:** India demonstration using Bangladesh-derived historical data, clearly labelled as synthetic.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── run_event_pipeline.py
│   ├── train_event_detection.py
│   ├── event_information_extraction.py
│   ├── country_resolver.py
│   ├── get_historical_forecast.py
│   ├── build_geonames_db.py
│   └── other scripts
├── models/
│   ├── event_detection_tfidf.joblib
│   └── protest_forecasting_model.joblib
├── data/
│   └── processed/
│       ├── daily_protest_forecasting.csv
│       └── india_synthetic_demo.csv
└── results/
    └── evaluation results
