# Path: Electric_Vehicle_Prediction/constants/__init__.py
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Define the project name
Project_Name = "Electric_Vehicle_Prediction"

# Define the list of constant to be created in the project

MONGODB_DATABASE_NAME = os.getenv("DATABASE_NAME")
MONGODB_COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MONGODB_URI = os.getenv("MONGODB_URI")

