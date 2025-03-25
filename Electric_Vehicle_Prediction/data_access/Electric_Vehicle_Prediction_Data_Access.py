# Path: Electric_Vehicle_Prediction/data_access/Electric_Vehicle_Prediction_Data_Access.py

from Electric_Vehicle_Prediction.configurations.mongo_db_connection import (
    mongodb_client,
)
from Electric_Vehicle_Prediction.constants import (
    MONGODB_DATABASE_NAME,
    MONGODB_COLLECTION_NAME,
)
from Electric_Vehicle_Prediction.exceptions import EV_Exception
import pandas as pd
import sys
import numpy as np


class EV_Dataframe:
    def __init__(self):
        try:
            self.client = mongodb_client
            self.database = self.client[MONGODB_DATABASE_NAME]
            self.collection = self.database[MONGODB_COLLECTION_NAME]
        except Exception as e:
            raise EV_Exception(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        try:
            data = self.collection.find({}, {"_id": 0, "index": 0})
            df = pd.DataFrame(list(data))
            df.drop_duplicates(inplace=True)
            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise EV_Exception(e, sys)
