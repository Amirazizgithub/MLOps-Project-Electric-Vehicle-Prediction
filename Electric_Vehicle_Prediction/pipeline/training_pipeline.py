import sys
from Electric_Vehicle_Prediction.exceptions import EV_Exception
from Electric_Vehicle_Prediction.logger import logging
from Electric_Vehicle_Prediction.components.data_ingestion import DataIngestion
from Electric_Vehicle_Prediction.components.data_validation import DataValidation
from Electric_Vehicle_Prediction.components.data_transformation import DataTransformation
from Electric_Vehicle_Prediction.components.model_trainer import ModelTrainer
# from Electric_Vehicle_Prediction.components.model_evaluation import ModelEvaluation
# from Electric_Vehicle_Prediction.components.model_pusher import ModelPusher


from Electric_Vehicle_Prediction.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    # ModelEvaluationConfig,
    # ModelPusherConfig,
)

from Electric_Vehicle_Prediction.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    # ModelEvaluationArtifact,
    # ModelPusherArtifact,
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        # self.model_evaluation_config = ModelEvaluationConfig()
        # self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info(
                "Entered the start_data_ingestion method of TrainPipeline class"
            )
            # data_ingestion = DataIngestion(
            #     data_ingestion_config=self.data_ingestion_config
            # )
            data_ingestion = DataIngestion()
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise EV_Exception(e, sys) from e

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logging.info("Entered the start_data_validation method of TrainPipeline class")
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact

        except Exception as e:
            raise EV_Exception(e, sys) from e

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        try:
            logging.info("Entered the start_data_transformation method of TrainPipeline class")
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact,
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise EV_Exception(e, sys)
        
    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact

        except Exception as e:
            raise EV_Exception(e, sys)
        
    def run_pipeline(
        self,
    ) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
            )
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )
            # model_evaluation_artifact = self.start_model_evaluation(
            #     data_ingestion_artifact=data_ingestion_artifact,
            #     model_trainer_artifact=model_trainer_artifact,
            # )

            # if not model_evaluation_artifact.is_model_accepted:
            #     logging.info(f"Model not accepted.")
            #     return None
            # model_pusher_artifact = self.start_model_pusher(
            #     model_evaluation_artifact=model_evaluation_artifact
            # )
            return model_trainer_artifact
        except Exception as e:
            raise EV_Exception(e, sys) from e
        
TrainPipeline().run_pipeline()