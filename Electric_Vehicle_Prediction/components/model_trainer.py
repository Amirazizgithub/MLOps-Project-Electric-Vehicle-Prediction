# Path: US_Visa_Prediction/components/model_trainer.py
import sys
from typing import Tuple
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from Electric_Vehicle_Prediction.exceptions import EV_Exception
from Electric_Vehicle_Prediction.logger import logging
from Electric_Vehicle_Prediction.utils.main_utils import main_utils
from Electric_Vehicle_Prediction.entity.config_entity import ModelTrainerConfig
from Electric_Vehicle_Prediction.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    RegressionMetricArtifact,
)


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(
        self, train: np.array, test: np.array
    ) -> Tuple[object, object]:
        try:
            logging.info("Using XGBoost to get best model object and report")

            x_train, y_train, x_test, y_test = (
                train[:, :-1],
                train[:, -1],
                test[:, :-1],
                test[:, -1],
            )
            XGBRegressor_model = XGBRegressor(max_depth=7, min_child_weight=1)
            best_model = XGBRegressor_model.fit(x_train, y_train)

            y_pred = best_model.predict(x_test)
            r_squared_score = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            model_accuracy = 1 - (rmse / np.mean(y_test))
            metric_artifact = RegressionMetricArtifact(
                r2_score=r_squared_score,
                mse=mse,
                rmse=rmse,
                accuracy=model_accuracy,
            )
            logging.info(f"Model report: {metric_artifact}")
            return best_model, metric_artifact, model_accuracy

        except Exception as e:
            raise EV_Exception(e, sys) from e

    def initiate_model_trainer(
        self,
    ) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            train_arr = main_utils.load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_training_file_path
            )
            test_arr = main_utils.load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_testing_file_path
            )

            best_model, metric_artifact, model_accuracy = (
                self.get_model_object_and_report(train=train_arr, test=test_arr)
            )

            if model_accuracy < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")

            main_utils.save_object(
                self.model_trainer_config.trained_model_file_path, best_model
            )

            main_utils.write_json_file(
                file_path=self.model_trainer_config.metric_artifact_file_path,
                content=metric_artifact.__dict__,
                replace=True,
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact_file_path=self.model_trainer_config.metric_artifact_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise EV_Exception(e, sys) from e
