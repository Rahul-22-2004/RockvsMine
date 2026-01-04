import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError(
        "MONGO_URI is not set. Check that .env is inside backend/ and correctly formatted."
    )

if not (MONGO_URI.startswith("mongodb://") or MONGO_URI.startswith("mongodb+srv://")):
    raise RuntimeError(f"Invalid MONGO_URI format: {MONGO_URI}")

client = MongoClient(MONGO_URI)

db = client["rockmine_db"]

users_collection = db["users"]
predictions_collection = db["predictions"]
