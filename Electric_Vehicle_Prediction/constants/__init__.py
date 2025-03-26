# Path: Electric_Vehicle_Prediction/constants/__init__.py
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Define the project name
Project_Name: str = "Electric_Vehicle_Prediction"

# Define the list of constant to be created in the project

MONGODB_DATABASE_NAME: str = os.getenv("DATABASE_NAME")
MONGODB_COLLECTION_NAME: str = os.getenv("COLLECTION_NAME")
MONGODB_URI: str = os.getenv("MONGODB_URI")

PIPELINE_NAME: str = "ev_pipeline"
ARTIFACTS_DIR: str = "artifacts"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

FILE_NAME: str = "EV_Data.csv"
MODEL_FILE_NAME = os.getenv("MODEL_FILE_NAME")


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "EV_Data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

