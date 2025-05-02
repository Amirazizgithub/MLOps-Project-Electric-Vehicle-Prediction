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
    transformed_object_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    drift_report_file_path: str

@dataclass
class RegressionMetricArtifact:
    r2_score: float
    mse: float
    rmse: float
    accuracy: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metric_artifact: RegressionMetricArtifact

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    changed_accuracy: float
    trained_model_path: str

