import sqlite3

DB_PATH = "data/external/geonames/geonames.db"


def get_country_iso3(country_code):
    """
    Convert ISO2 country code to ISO3.
    Uses pycountry for complete country coverage.
    """
    import pycountry

    country = pycountry.countries.get(alpha_2=country_code)

    if country:
        return country.alpha_3

    return None
def resolve_country(locations):
    connection = sqlite3.connect(DB_PATH)

    candidates = {}

    for location in locations:
        location = location.strip()

        if not location:
            continue

        # Search exact city names and alternate names
        
        # First: search exact city names and ASCII names
        exact_query = """
        SELECT name, country_code, admin1, population
        FROM cities
        WHERE name = ? COLLATE NOCASE
           OR asciiname = ? COLLATE NOCASE
        ORDER BY CAST(population AS INTEGER) DESC
        LIMIT 100
        """

        rows = connection.execute(
            exact_query, (location, location)
        ).fetchall()

        # Only search alternate names if no exact match exists
        if not rows:
            alternate_query = """
            SELECT name, country_code, admin1, population
            FROM cities
            WHERE instr(
                ',' || lower(alternatenames) || ',',
                ',' || lower(?) || ','
            ) > 0
            ORDER BY CAST(population AS INTEGER) DESC
            LIMIT 100
            """

            rows = connection.execute(
                alternate_query, (location,)
            ).fetchall()

        for name, country, admin1, population in rows:
            iso3 = get_country_iso3(country)

            if iso3:
                candidates.setdefault(iso3, []).append({
                    "matched_location": location,
                    "matched_name": name,
                    "admin1": admin1,
                    "population": int(population or 0)
                })

    connection.close()

    if not candidates:
        return {
            "country_code": None,
            "status": "unresolved"
        }

    # Find the candidate with the highest population
    best_country = max(
        candidates,
        key=lambda country: max(
            item["population"] for item in candidates[country]
        )
    )

    best_evidence = max(
        candidates[best_country],
        key=lambda item: item["population"]
    )

    return {
        "country_code": best_country,
        "status": "resolved",
        "evidence": [best_evidence],
        "all_candidates": candidates
    }


if __name__ == "__main__":
    tests = [
        ["Mumbai", "India"],
        ["Pune"],
        ["Jaipur"],
        ["Dhaka"],
        ["Springfield"],
        ["UnknownPlace123"]
    ]

    for locations in tests:
        print("\nInput:", locations)
        print(resolve_country(locations))
