# Path: US_Visa_Prediction/pipeline/prediction_pipeline.py
import sys

import mlflow
import mlflow.sklearn
from typing import Optional
from pandas import DataFrame
from Electric_Vehicle_Prediction.entity.config_entity import ModelRegisterConfig, DataTransformationConfig
from Electric_Vehicle_Prediction.exceptions import EV_Exception
from Electric_Vehicle_Prediction.utils.main_utils import main_utils
from Electric_Vehicle_Prediction.logger import logging
from pandas import DataFrame

class ElectricVehicleData:
    def __init__(
        self,
        model_year,
        make,
        model,
        electric_vehicle_type,
        clean_alternative_fuel_vehicle_eligibility,
        base_msrp,
    ):
        try:
            self.model_year = model_year
            self.make = make
            self.model = model
            self.electric_vehicle_type = electric_vehicle_type
            self.clean_alternative_fuel_vehicle_eligibility = clean_alternative_fuel_vehicle_eligibility
            self.base_msrp = base_msrp

        except Exception as e:
            raise EV_Exception(e, sys) from e
        
    def get_ev_data_as_dict(self) -> dict:
        try:
            input_data = {
                "Model Year": [self.model_year],
                "Make": [self.make],
                "Model": [self.model],
                "Electric Vehicle Type": [self.electric_vehicle_type],
                "Clean Alternative Fuel Vehicle (CAFV) Eligibility": [self.clean_alternative_fuel_vehicle_eligibility],
                "Base MSRP": [self.base_msrp],
            }

            logging.info("Created electric vehicle input data dict from api")

            return input_data

        except Exception as e:
            raise EV_Exception(e, sys) from e

    def get_ev_input_data_frame(self) -> DataFrame:
        try:

            ev_input_dict = self.get_ev_data_as_dict()
            logging.info("Converted electric vehicle input data dict to dataframe")
            return DataFrame(ev_input_dict)

        except Exception as e:
            raise EV_Exception(e, sys) from e


class EVPredictor:
    def __init__(
        self,
        model_register_config: ModelRegisterConfig = ModelRegisterConfig(),
        data_transformation_config: DataTransformationConfig = DataTransformationConfig(),
    ) -> float:
        try:
            # self.schema_config = main_utils.read_yaml_file(SCHEMA_FILE_PATH)
            self.model_register_config = model_register_config
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise EV_Exception(e, sys)
        
    def load_mlflow_production_model(self) -> Optional[object]:
        logging.info("Start Loading of model from MLflow")
        # Set tracking URI and experiment name
        if not self.model_register_config.ec2_mlflow_tracking_uri:
            raise ValueError("EC2_MLFLOW_TRACKING_URI environment variable is not set.")

        mlflow.set_tracking_uri(self.model_register_config.ec2_mlflow_tracking_uri)
        mlflow.set_experiment(self.model_register_config.experiment_name)

        try:
            # Create an MlflowClient to interact with the MLflow server
            client = mlflow.tracking.MlflowClient()

            # Get the latest version of the model in the Production stage
            model_name = self.model_register_config.mlflow_model_name
            versions = client.get_latest_versions(model_name, stages=["production"])

            if versions:
                latest_version = versions[0].version
                run_id = versions[
                    0
                ].run_id  # Fetching the run ID from the latest version
                logging.info(
                    f"Latest version in Production: {latest_version}, Run ID: {run_id}"
                )

                # Construct the logged model path
                logged_model = f"runs:/{run_id}/model"

                # Load the model
                loaded_model = mlflow.sklearn.load_model(logged_model)
                logging.info(f"Model loaded from {logged_model}")

                return loaded_model
            else:
                logging.info("No model found in the 'Production' stage on MLflow.")
                return None

        except Exception as e:
            raise EV_Exception(e, sys) from e
        
    # Load Preprocessor from Artifact Store
    def load_preprocessor(self) -> Optional[object]:
        try:
            # Load the preprocessor from the artifact store
            preprocessor = main_utils.save_object(
                    self.data_transformation_config.transformed_object_file_path,
                    preprocessor,
                )
            logging.info("Preprocessor loaded successfully")

            return preprocessor

        except Exception as e:
            raise EV_Exception(e, sys) from e

    def predict(self, dataframe) -> float:
        try:
            logging.info("Entered predict method of USvisaClassifier class")
            model = self.load_mlflow_production_model()
            if model is None:
                raise ValueError("Model not found in the 'Production' stage on MLflow.")
            preprocessor = self.load_preprocessor()
            if preprocessor is None:
                raise ValueError("Preprocessor not found in the artifact store.")
            
            try:
                preprocessed_data = preprocessor.transform(dataframe)
                result = model.predict(preprocessed_data)

                return result
            
            except Exception as e:
                raise EV_Exception(e, sys) from e

        except Exception as e:
            raise EV_Exception(e, sys) from e
        
