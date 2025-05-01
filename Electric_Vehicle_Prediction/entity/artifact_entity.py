# Path: Electric_Vehicle_Prediction/entity/artifact_entity.py

from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    training_file_path: str
    testing_file_path: str
    feature_store_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_training_file_path: str
    transformed_testing_file_path: str
    preprocessed_object_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    drift_report_file_path: str

