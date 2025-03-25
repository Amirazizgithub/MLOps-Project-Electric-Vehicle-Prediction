# Path: Electric_Vehicle_Prediction/configurations/mongo_db_connection.py

import sys
from pymongo import MongoClient
from Electric_Vehicle_Prediction.constants import MONGODB_DATABASE_NAME, MONGODB_URI
from Electric_Vehicle_Prediction.logger import logging
from Electric_Vehicle_Prediction.exceptions import USvisaException
import certifi

ca = certifi.where()


class MongoDBClient:
    client = None

    def __init__(self, database_name=MONGODB_DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = MONGODB_URI
                if mongo_db_url is None:
                    raise Exception(f"Environment key: {MONGODB_URI} is not set.")
                MongoDBClient.client = MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection succesfull")
        except Exception as e:
            raise USvisaException(e, sys)

    def mongodb_collections(self):
        return self.database.list_collection_names()


col_name = MongoDBClient().mongodb_collections()

