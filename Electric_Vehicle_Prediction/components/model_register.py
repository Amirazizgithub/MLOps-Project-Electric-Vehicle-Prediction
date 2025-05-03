import sys
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from Electric_Vehicle_Prediction.utils.main_utils import main_utils
from Electric_Vehicle_Prediction.exceptions import EV_Exception
from Electric_Vehicle_Prediction.logger import logging
from Electric_Vehicle_Prediction.entity.config_entity import ModelRegisterConfig
from Electric_Vehicle_Prediction.entity.artifact_entity import ModelTrainerArtifact


class ModelPusher:
    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_register_config: ModelRegisterConfig,
    ):
        try:
            self.model_trainer_artifact = model_trainer_artifact
            self.model_register_config = model_register_config
        except Exception as e:
            raise EV_Exception(e, sys) from e

    def initiate_model_register_to_mlflow(self):
        try:
            logging.info("Start Registeration of model to MLflow")
            # Set tracking URI and experiment name
            if not self.model_register_config.ec2_mlflow_tracking_uri:
                raise ValueError(
                    "EC2_MLFLOW_TRACKING_URI environment variable is not set."
                )

            mlflow.set_tracking_uri(self.model_register_config.ec2_mlflow_tracking_uri)
            mlflow.set_experiment(self.model_register_config.experiment_name)

            # Start MLflow run
            with mlflow.start_run(run_name="XGBoost_Model"):
                logging.info("Started MLflow run")
                logging.info("Logging model metrics to MLflow")

                # Log Model Metrics
                metric_artifact = main_utils.read_json_file(
                    file_path=self.model_trainer_artifact.metric_artifact_file_path
                )
                mlflow.log_metric("r2_score", metric_artifact["r2_score"])
                mlflow.log_metric("mse", metric_artifact["mse"])
                mlflow.log_metric("rmse", metric_artifact["rmse"])
                mlflow.log_metric("accuracy", metric_artifact["accuracy"])

                logging.info("Logging model to MLflow")
                # Log Model
                model = main_utils.load_object(
                    self.model_trainer_artifact.trained_model_file_path
                )
                mlflow.sklearn.log_model(model, "model")
                mlflow.log_param("model_type", "XGBoost")
                mlflow.log_param("max_depth", 7)
                mlflow.log_param("min_child_weight", 1)
                logging.info("Model registered to MLflow")

                # Get run ID
                run_id = mlflow.active_run().info.run_id

            # Register Model
            client = MlflowClient()
            model_name = self.model_register_config.mlflow_model_name

            # Check if the model is already registered
            try:
                client.get_registered_model(model_name)
                logging.info(f"Model '{model_name}' already registered.")
            except mlflow.exceptions.RestException:
                client.create_registered_model(model_name)
                logging.info(f"Registered new model: {model_name}")

            # Create a new model version
            artifact_uri = f"runs:/{run_id}/model"
            model_version = client.create_model_version(
                name=model_name, source=artifact_uri, run_id=run_id
            )
            logging.info(f"Created model version: {model_version.version}")

            # Transition the model version to Production
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production",
                archive_existing_versions=True,
            )
            logging.info(
                f"Model version {model_version.version} transitioned to Production stage"
            )

            logging.info(f"Model successfully pushed to MLflow. Run ID: {run_id}")

        except Exception as e:
            raise EV_Exception(e, sys) from e
