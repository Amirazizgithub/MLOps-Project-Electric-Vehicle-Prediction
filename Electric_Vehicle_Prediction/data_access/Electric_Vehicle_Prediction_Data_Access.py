# Path: Electric_Vehicle_Prediction/data_access/Electric_Vehicle_Prediction_Data_Access.py
import pandas as pd
import sys
import numpy as np
from Electric_Vehicle_Prediction.exceptions import EV_Exception
from Electric_Vehicle_Prediction.logger import logging
from Electric_Vehicle_Prediction.constants import SCHEMA_FILE_PATH
from Electric_Vehicle_Prediction.utils.main_utils import main_utils
from Electric_Vehicle_Prediction.configurations.mongo_db_connection import (
    mongodb_client,
)
from Electric_Vehicle_Prediction.constants import (
    MONGODB_DATABASE_NAME,
    MONGODB_COLLECTION_NAME,
)


class EV_Dataframe:
    def __init__(self):
        try:
            self.client = mongodb_client
            self.database = self.client[MONGODB_DATABASE_NAME]
            self.collection = self.database[MONGODB_COLLECTION_NAME]
            self._schema_config = main_utils.read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            raise EV_Exception(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        try:
            logging.info("Exporting collection as dataframe")
            data = self.collection.find({}, {"_id": 0, "index": 0})
            df = pd.DataFrame(list(data))
            drop_columns = self._schema_config["drop_columns"]
            df.drop(columns=drop_columns, inplace=True)
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            raise EV_Exception(e, sys)
