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

TARGET_COLUMN = "Electric Range"
CURRENT_YEAR = datetime.today().year
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "EV_Data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

