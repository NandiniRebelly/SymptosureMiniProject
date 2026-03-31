from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["symptom_checker"]

users_collection = db["users"]

history_collection = db["history"]