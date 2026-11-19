import spacy

nlp = spacy.load("en_core_web_sm")

tweets = [
    "Farmers will gather at India Gate, New Delhi, on 25 November at 10 AM to protest against the new agricultural policy.",
    "Students from Bennett University will organise a demonstration in Greater Noida tomorrow.",
    "Rahul Gandhi addressed protesters in Delhi demanding justice for affected workers."
]

for tweet in tweets:
    doc = nlp(tweet)

    print("\nTweet:", tweet)
    print("-" * 60)

    for ent in doc.ents:
        print(f"{ent.text:30} {ent.label_:15} {spacy.explain(ent.label_)}")
