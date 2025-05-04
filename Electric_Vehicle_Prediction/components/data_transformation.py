# Path: US_Visa_Prediction/components/data_transformation.py
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.compose import ColumnTransformer

from Electric_Vehicle_Prediction.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from Electric_Vehicle_Prediction.entity.config_entity import DataTransformationConfig
from Electric_Vehicle_Prediction.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
)
from Electric_Vehicle_Prediction.exceptions import EV_Exception
from Electric_Vehicle_Prediction.logger import logging
from Electric_Vehicle_Prediction.utils.main_utils import main_utils


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = main_utils.read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise EV_Exception(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise EV_Exception(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ordinal_encoder = OrdinalEncoder()

            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            oh_columns = self._schema_config["oh_columns"]
            or_columns = self._schema_config["or_columns"]
            num_features = self._schema_config["num_features"]

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("StandardScaler", numeric_transformer, num_features),
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")
            return preprocessor

        except Exception as e:
            raise EV_Exception(e, sys) from e

    def initiate_data_transformation(
        self,
    ) -> DataTransformationArtifact:
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(
                    file_path=self.data_ingestion_artifact.training_file_path
                )
                test_df = DataTransformation.read_data(
                    file_path=self.data_ingestion_artifact.testing_file_path
                )

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info(
                    "Got input features and target features of Training dataset"
                )

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)

                target_feature_test_df = test_df[TARGET_COLUMN]

                logging.info(
                    "Got input features and target features of Testing dataset"
                )

                logging.info(
                    "Applying preprocessing object on training dataframe and testing dataframe"
                )

                input_feature_train_arr = preprocessor.fit_transform(
                    input_feature_train_df
                )

                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                logging.info(
                    "Used the preprocessor object to transform the train & test features"
                )

                logging.info("Created train array and test array")

                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]

                test_arr = np.c_[
                    input_feature_test_arr, np.array(target_feature_test_df)
                ]

                main_utils.save_object(
                    self.data_transformation_config.transformed_object_file_path,
                    preprocessor,
                )
                main_utils.save_numpy_array_data(
                    self.data_transformation_config.transformed_training_file_path,
                    array=train_arr,
                )
                main_utils.save_numpy_array_data(
                    self.data_transformation_config.transformed_testing_file_path,
                    array=test_arr,
                )

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_training_file_path=self.data_transformation_config.transformed_training_file_path,
                    transformed_testing_file_path=self.data_transformation_config.transformed_testing_file_path,
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise EV_Exception(e, sys) from e
        
