# Path: Electric_Vehicle_Prediction/components/data_validation.py

import json
import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from pandas import DataFrame

from Electric_Vehicle_Prediction.exceptions import EV_Exception
from Electric_Vehicle_Prediction.logger import logging
from Electric_Vehicle_Prediction.utils.main_utils import main_utils
from Electric_Vehicle_Prediction.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from Electric_Vehicle_Prediction.entity.config_entity import DataValidationConfig
from Electric_Vehicle_Prediction.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = main_utils.read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise EV_Exception(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        try:
            status = len(dataframe.columns) == len(
                self._schema_config["remaining_columns"]
            )
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise EV_Exception(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns) > 0:
                logging.info(
                    f"Missing categorical column: {missing_categorical_columns}"
                )

            return (
                False
                if len(missing_categorical_columns) > 0
                or len(missing_numerical_columns) > 0
                else True
            )
        except Exception as e:
            raise EV_Exception(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise EV_Exception(e, sys)

    def detect_dataset_drift(
        self,
        reference_df: DataFrame,
        current_df: DataFrame,
    ) -> bool:
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_df, current_data=current_df)
            data_drift_status = report.include_tests

            json_report = {"data_drift_status": data_drift_status}
            main_utils.write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=json_report,
            )

            return data_drift_status

        except Exception as e:
            raise EV_Exception(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df, test_df = (
                DataValidation.read_data(
                    file_path=self.data_ingestion_artifact.training_file_path
                ),
                DataValidation.read_data(
                    file_path=self.data_ingestion_artifact.testing_file_path
                ),
            )

            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(
                f"All required columns present in training dataframe: {status}"
            )
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."

            status = self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."

            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."

            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += f"columns are missing in test dataframe."

            validation_status = len(validation_error_msg) == 0

            if validation_status:
                data_drift_status = self.detect_dataset_drift(train_df, test_df)
                if data_drift_status:
                    logging.info(f"Drift Detected.")
                    validation_error_msg = "Drift Detected"
                else:
                    validation_error_msg = "Drift Not Detected"
            else:
                logging.info(f"Validation_error: {validation_error_msg}")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise EV_Exception(e, sys) from e
