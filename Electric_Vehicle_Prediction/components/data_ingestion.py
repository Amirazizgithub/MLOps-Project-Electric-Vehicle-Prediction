# Path: Electric_Vehicle_Prediction/components/data_ingestion.py
import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from Electric_Vehicle_Prediction.entity.config_entity import DataIngestionConfig
from Electric_Vehicle_Prediction.entity.artifact_entity import DataIngestionArtifact
from Electric_Vehicle_Prediction.exceptions import EV_Exception
from Electric_Vehicle_Prediction.logger import logging
from Electric_Vehicle_Prediction.data_access.Electric_Vehicle_Prediction_Data_Access import (
    EV_Dataframe,
)


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.ev_dataframe = EV_Dataframe()

    def export_data_into_feature_store(self) -> DataFrame:
        try:
            logging.info(f"Exporting data from mongodb")
            dataframe = self.ev_dataframe.export_collection_as_dataframe()
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            # Create folder
            dir_path = os.path.dirname(feature_store_file_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except EV_Exception as e:
            logging.error(f"Error occurred: {e}")
            raise EV_Exception(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
            )
            logging.info("Data split into train and test sets")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

            logging.info(f"Exported train and test file path.")

        except EV_Exception as e:
            logging.error(f"Error occurred: {e}")
            raise EV_Exception(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Got the data from mongodb")

            self.split_data_as_train_test(dataframe)
            logging.info("Performed train test split on the dataset")

            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                training_file_path=self.data_ingestion_config.training_file_path,
                testing_file_path=self.data_ingestion_config.testing_file_path,
            )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except EV_Exception as e:
            logging.error(f"Error occurred: {e}")
            raise EV_Exception(e, sys)


# dataframe=DataIngestion().export_data_into_feature_store()
# DataIngestion().split_data_as_train_test(dataframe)
# DataIngestion().initiate_data_ingestion()
