# Path: Electric_Vehicle_Prediction/configurations/mongo_db_connection.py

import sys
from pymongo import MongoClient
from Electric_Vehicle_Prediction.constants import MONGODB_DATABASE_NAME, MONGODB_URI
from Electric_Vehicle_Prediction.logger import logging
from Electric_Vehicle_Prediction.exceptions import EV_Exception
import certifi

ca = certifi.where()


class MongoDB_Client:
    client = None

    def __init__(self, database_name=MONGODB_DATABASE_NAME) -> None:
        try:
            if MongoDB_Client.client is None:
                mongo_db_url = MONGODB_URI
                if mongo_db_url is None:
                    raise Exception(f"Environment key: {MONGODB_URI} is not set.")
                MongoDB_Client.client = MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDB_Client.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection succesfull")
        except Exception as e:
            raise EV_Exception(e, sys)

    def get_client(self) -> MongoClient:
        return self.client

    def mongodb_collections(self):
        return self.database.list_collection_names()


mongodb_client = MongoDB_Client().get_client()
