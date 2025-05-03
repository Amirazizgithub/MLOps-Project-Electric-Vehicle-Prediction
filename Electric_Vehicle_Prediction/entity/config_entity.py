# Path: Electric_Vehicle_Prediction/entity/config_entity.py

import os
from Electric_Vehicle_Prediction.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


# Define the training pipeline config
@dataclass
class TrainingPipelineconfig:
    pipeline_name: str = PIPELINE_NAME
    ARTIFACTS_DIR: str = os.path.join(ARTIFACTS_DIR, TIMESTAMP)
    MODEL_FILE_NAME: str = MODEL_FILE_NAME
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineconfig = TrainingPipelineconfig()


# Define the data ingestion config
@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.ARTIFACTS_DIR, DATA_INGESTION_DIR_NAME
    )
    feature_store_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME
    )
    training_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME
    )
    testing_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME
    )
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME


# Define the data validation config
@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_pipeline_config.ARTIFACTS_DIR, DATA_VALIDATION_DIR_NAME
    )
    drift_report_file_path: str = os.path.join(
        data_validation_dir,
        DATA_VALIDATION_DRIFT_REPORT_DIR,
        DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
    )


# Define the data transformation config
@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(
        training_pipeline_config.ARTIFACTS_DIR, DATA_TRANSFORMATION_DIR_NAME
    )
    transformed_train_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        TRAIN_FILE_NAME.replace("csv", "npy"),
    )
    transformed_test_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        TEST_FILE_NAME.replace("csv", "npy"),
    )
    transformed_object_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
        PREPROCSSING_OBJECT_FILE_NAME,
    )


# Define the model trainer config
@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.ARTIFACTS_DIR, MODEL_TRAINER_DIR_NAME
    )
    trained_model_file_path: str = os.path.join(
        model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME
    )
    metric_artifact_file_path: str = os.path.join(
        model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_METRIC_FILE_NAME
    )
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE


@dataclass
class ModelRegisterConfig:
    ec2_mlflow_tracking_uri: str = EC2_MLFLOW_TRACKING_URI
    experiment_name: str = EXPERIMENT_NAME
    mlflow_model_name: str = MLFLOW_MODEL_NAME


# Define the model evaluation config
@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
