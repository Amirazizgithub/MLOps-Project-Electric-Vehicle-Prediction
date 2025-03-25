import pandas as pd
from pymongo import MongoClient
from Electric_Vehicle_Prediction.constants import (
    MONGODB_DATABASE_NAME,
    MONGODB_COLLECTION_NAME,
)
import os
from dotenv import load_dotenv

load_dotenv()

# Define the class to interact with the MongoDB database and store the data in the collection
class MongoDBConnection:
    def __init__(self):
        self.client = MongoClient(
            os.getenv("MONGODB_URI"),
            maxPoolSize=50,
            socketTimeoutMS=50000,
            connectTimeoutMS=50000,
        )
        self.db = self.client[MONGODB_DATABASE_NAME]
        self.collection = self.db[MONGODB_COLLECTION_NAME]
        self.batch_size = 20000

    def insert_data(self, batch_size=1000) -> str:
        # Load the data from the CSV file
        data = pd.read_csv("data/Electric_Vehicle_Data.csv")
        data.reset_index(inplace=True)

        # Convert the data to a dictionary
        data_dict = data.to_dict("records")

        # Insert the data in batches
        for i in range(0, len(data_dict), batch_size):
            batch = data_dict[i : i + batch_size]
            self.collection.insert_many(batch)

        return "Data inserted successfully!"

# Insert data into MongoDB
MongoDBConnection().insert_data()

'''The error indicates that the connection to MongoDB is being closed, likely due to the large size of the dataset (2,32,230 records) 
being inserted in a single operation. To resolve this, you can batch the insertion process into smaller chunks to avoid overwhelming 
the MongoDB server.'''

