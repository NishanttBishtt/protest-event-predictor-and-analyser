import joblib

model = joblib.load("models/event_detection_tfidf.joblib")

tweets = [
    "Thousands of people gathered in Delhi to protest against the new government policy.",
    "The students are organising a demonstration outside the university tomorrow.",
    "Bitcoin prices increased significantly this morning.",
    "Enjoy 50 percent off on all products. Shop now!",
    "Workers are planning a march demanding higher wages and better working conditions."
]

predictions = model.predict(tweets)
probabilities = model.predict_proba(tweets)

for tweet, prediction, probability in zip(
    tweets, predictions, probabilities
):
    label = "Event" if prediction == 1 else "Non-event"

    print("\nTweet:", tweet)
    print("Prediction:", label)
    print("Event probability:", round(probability[1], 4))
