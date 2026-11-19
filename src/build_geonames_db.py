
import sqlite3
import pandas as pd

DATA_DIR = "data/external/geonames"
DB_PATH = "data/external/geonames/geonames.db"

columns = [
    "geonameid", "name", "asciiname", "alternatenames",
    "latitude", "longitude", "feature_class", "feature_code",
    "country_code", "cc2", "admin1", "admin2", "admin3",
    "admin4", "population", "elevation", "dem",
    "timezone", "modification_date"
]

print("Reading GeoNames cities...")
cities = pd.read_csv(
    f"{DATA_DIR}/cities500.txt",
    sep="\t",
    names=columns,
    dtype=str,
    keep_default_na=False
)

print("Total geographical records:", len(cities))

connection = sqlite3.connect(DB_PATH)

cities.to_sql("cities", connection, if_exists="replace", index=False)

connection.execute("""
CREATE INDEX IF NOT EXISTS idx_city_name
ON cities(name COLLATE NOCASE)
""")

connection.execute("""
CREATE INDEX IF NOT EXISTS idx_city_country
ON cities(country_code)
""")

connection.execute("""
CREATE INDEX IF NOT EXISTS idx_city_admin
ON cities(country_code, admin1)
""")

connection.commit()
connection.close()

print("GeoNames database created:", DB_PATH)
