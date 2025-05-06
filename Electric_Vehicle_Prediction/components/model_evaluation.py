from Electric_Vehicle_Prediction.entity.config_entity import ModelRegisterConfig
from Electric_Vehicle_Prediction.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
)
from sklearn.metrics import mean_squared_error
from Electric_Vehicle_Prediction.exceptions import EV_Exception
from Electric_Vehicle_Prediction.logger import logging
from Electric_Vehicle_Prediction.utils.main_utils import main_utils
import sys
import numpy as np
import pandas as pd
from typing import Tuple
import mlflow
import mlflow.sklearn
from typing import Optional


class ModelEvaluation:

    def __init__(
        self,
        model_register_config: ModelRegisterConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> bool:
        try:
            self.model_register_config = model_register_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise EV_Exception(e, sys) from e
        
    def load_testing_data(self) -> np.array:
        try:
            test_arr = main_utils.load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_testing_file_path
            )
            logging.info(f"Testing data loaded from {self.data_transformation_artifact.transformed_testing_file_path}")
            return test_arr
        except Exception as e:
            raise EV_Exception(e, sys) from e
        
    def load_new_trained_model(self) -> object:
        try:
            trained_model = main_utils.load_object(
                file_path=self.model_trainer_artifact.trained_model_file_path
            )
            logging.info(f"Newly trained model loaded from {self.model_trainer_artifact.trained_model_file_path}")
            return trained_model
        except Exception as e:
            raise EV_Exception(e, sys) from e
        
    def load_mlflow_production_model(self) -> Optional[object]:
        logging.info("Start Loading of model from MLflow")        
        # Set tracking URI and experiment name
        if not self.model_register_config.ec2_mlflow_tracking_uri:
            raise ValueError(
                "EC2_MLFLOW_TRACKING_URI environment variable is not set."
            )

        mlflow.set_tracking_uri(self.model_register_config.ec2_mlflow_tracking_uri)

        try:
            # Create an MlflowClient to interact with the MLflow server
            client = mlflow.tracking.MlflowClient()

            # Get the latest version of the model in the Production stage
            model_name = self.model_register_config.mlflow_model_name
            versions = client.get_latest_versions(model_name, stages=["production"])

            if versions:
                latest_version = versions[0].version
                run_id = versions[0].run_id  # Fetching the run ID from the latest version
                logging.info(f"Latest version in Production: {latest_version}, Run ID: {run_id}")

                # Construct the logged model path
                logged_model = f"runs:/{run_id}/model"

                # Load the model
                loaded_model = mlflow.sklearn.load_model(logged_model)
                logging.info(f"Model loaded from {logged_model}")

                return loaded_model
            else:
                logging.info("No model found in the 'Production' stage.")
                return None

        except Exception as e:
            raise EV_Exception(e, sys) from e
        
    def get_model_evaluation_object_and_report(
        self, model: object, test_df: np.array
    ) -> Tuple[object, float]:
        try:
            logging.info("Starting model evaluation")
            x_test, y_test = test_df[:, :-1], test_df[:, -1]

            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            model_accuracy = 1 - (rmse / np.mean(y_test))
            logging.info(f"Model report: {model_accuracy}")
            return model, model_accuracy

        except Exception as e:
            raise EV_Exception(e, sys) from e
        
    def initiate_model_evaluation(self) -> bool:
        try:
            logging.info("Entered initiate_model_evaluation method of ModelEvaluation class")
            test_arr = self.load_testing_data()
            new_trained_model = self.load_new_trained_model()
            mlflow_production_model = self.load_mlflow_production_model()

            if mlflow_production_model is None:
                logging.info("No model found in the 'Production' stage on MLFlow")
                return True  # This means we can register the new model if no production model exists on MLflow

            new_trained_model, new_trained_model_accuracy = self.get_model_evaluation_object_and_report(
                model=new_trained_model, test_df=test_arr
            )
            logging.info(f"New trained model accuracy: {new_trained_model_accuracy}")

            mlflow_production_model, mlflow_production_model_accuracy = self.get_model_evaluation_object_and_report(
                model=mlflow_production_model, test_df=test_arr
            )
            logging.info(f"MLflow production model accuracy: {mlflow_production_model_accuracy}")

            difference = (new_trained_model_accuracy - mlflow_production_model_accuracy)

            if difference > 0.05:
                logging.info(f"New model is better than the existing model. Differrence in accuracy: {difference}")
                return True
            else:
                logging.info(f"New model is not better than the existing model. Difference in accuracy: {difference}")
                return False

        except Exception as e:
            raise EV_Exception(e, sys) from e
