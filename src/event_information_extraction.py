
import spacy
import dateparser
import re
import json
import html

from datetime import datetime

nlp = spacy.load("en_core_web_sm")


# --------------------------------------------------
# 1. TEXT PREPROCESSING
# --------------------------------------------------

def clean_tweet(tweet):
    text = html.unescape(str(tweet))

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove usernames
    text = re.sub(r"@\w+", "", text)

    # Preserve hashtag words
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# 2. EVENT TYPE IDENTIFICATION
# --------------------------------------------------

def identify_event_type(tweet):

    event_types = {
        "protest": [
            "protest", "protests", "protesting",
            "protested", "protester", "protesters",
            "protestor", "protestors"
        ],
        "demonstration": [
            "demonstration", "demonstrations",
            "demonstrate", "demonstrating"
        ],
        "march": ["march", "marches", "marching"],
        "rally": ["rally", "rallies"],
        "strike": ["strike", "strikes", "striking", "walkout"]
    }

    text = tweet.lower()

    for event_type, keywords in event_types.items():
        for word in keywords:
            if re.search(r"\b" + re.escape(word) + r"\b", text):
                return event_type

    return None


# --------------------------------------------------
# 3. REMOVE DUPLICATES WHILE PRESERVING ORDER
# --------------------------------------------------

def unique_entities(items):

    result = []
    seen = set()

    for item in items:
        item = item.strip()

        if item and item.lower() not in seen:
            result.append(item)
            seen.add(item.lower())

    return result


# --------------------------------------------------
# 4. EVENT INFORMATION EXTRACTION
# --------------------------------------------------

def extract_event_information(tweet, reference_date=None):

    cleaned_tweet = clean_tweet(tweet)

    event_type = identify_event_type(cleaned_tweet)
    doc = nlp(cleaned_tweet)

    people = []
    organisations = []
    locations = []
    venues = []
    dates = []
    times = []

    # Named Entity Recognition
    for ent in doc.ents:

        if ent.label_ == "PERSON":
            people.append(ent.text)

        elif ent.label_ == "ORG":
            generic_groups = {
                "farmers", "students", "workers",
                "activists", "residents", "employees",
                "protesters", "protestors", "demonstrators",
                "teachers", "drivers", "unions"
            }

            if ent.text.lower().strip() not in generic_groups:
                organisations.append(ent.text)

        elif ent.label_ in ["GPE", "LOC"]:
            locations.append(ent.text)

        elif ent.label_ == "FAC":
            venues.append(ent.text)

        elif ent.label_ == "DATE":
            dates.append(ent.text)

        elif ent.label_ == "TIME":
            times.append(ent.text)

    # --------------------------------------------------
    # 5. GROUP EXTRACTION
    # --------------------------------------------------

    group_patterns = [
        r"\bfarmers?\b",
        r"\bstudents?\b",
        r"\bworkers?\b",
        r"\bactivists?\b",
        r"\bresidents?\b",
        r"\bemployees?\b",
        r"\bprotesters?\b",
        r"\bprotestors?\b",
        r"\bdemonstrators?\b",
        r"\bteachers?\b",
        r"\bdrivers?\b",
        r"\bunions?\b"
    ]

    groups = []

    for pattern in group_patterns:
        matches = re.findall(
            pattern, cleaned_tweet, re.IGNORECASE
        )
        groups.extend(matches)

    # --------------------------------------------------
    # 6. DATE NORMALISATION
    # --------------------------------------------------

    normalized_dates = []

    if isinstance(reference_date, str):
        reference_date = datetime.fromisoformat(reference_date)

    for date_text in dates:

        # Do not invent a year for dates without one.
        has_year = bool(re.search(r"\b20\d{2}\b", date_text))

        # Relative dates require the tweet's publication date.
        if reference_date is None and not has_year:
            continue

        settings = {}

        if reference_date is not None:
            settings["RELATIVE_BASE"] = reference_date

        parsed = dateparser.parse(
            date_text,
            settings=settings
        )

        if parsed:
            normalized_dates.append(
                parsed.strftime("%Y-%m-%d")
            )

    # --------------------------------------------------
    # 7. PROTEST CAUSE EXTRACTION
    # --------------------------------------------------

    cause = None

    cause_patterns = [
        r"protest(?:ing)?\s+against\s+(.+?)(?:[.,;!?]|$)",
        r"demonstrat(?:e|ing)\s+against\s+(.+?)(?:[.,;!?]|$)",
        r"demanding\s+(.+?)(?:[.,;!?]|$)",
        r"calling\s+for\s+(.+?)(?:[.,;!?]|$)",
        r"to\s+(?:save|protect|defend|support)\s+(.+?)(?:[.,;!?]|$)",
        r"against\s+(.+?)(?:[.,;!?]|$)"
    ]

    for pattern in cause_patterns:

        match = re.search(
            pattern, cleaned_tweet, re.IGNORECASE
        )

        if match:
            cause = match.group(1).strip()
            break

    # --------------------------------------------------
    # 8. STRUCTURED OUTPUT
    # --------------------------------------------------

    result = {
        "event_type": event_type,
        "original_tweet": tweet,
        "people": unique_entities(people),
        "organisations": unique_entities(organisations),
        "groups": unique_entities(groups),
        "locations": unique_entities(locations),
        "venues": unique_entities(venues),
        "dates": unique_entities(dates),
        "normalized_dates": unique_entities(normalized_dates),
        "times": unique_entities(times),
        "protest_cause": cause
    }

    return result


# --------------------------------------------------
# 9. TESTING
# --------------------------------------------------

if __name__ == "__main__":

    test_tweets = [
        "Farmers will gather at India Gate, New Delhi, on 25 November at 10 AM to protest against the new agricultural policy.",

        "Students from Bennett University will organise a demonstration in Greater Noida tomorrow.",

        "Rahul Gandhi addressed protesters in Delhi demanding justice for affected workers.",

        "If Gyanesh Kumar does not resign by tomorrow, CJP will begin its nationwide protest from Mumbai on 2 October, and take it to cities across India. This Gandhi Jayanti let's pledge to save democracy."
    ]

    for tweet in test_tweets:

        result = extract_event_information(tweet)

        print("\n" + "=" * 70)
        print(json.dumps(result, indent=4, ensure_ascii=False))
