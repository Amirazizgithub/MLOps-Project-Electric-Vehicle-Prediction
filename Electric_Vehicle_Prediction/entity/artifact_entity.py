# Path: Electric_Vehicle_Prediction/entity/artifact_entity.py

from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    training_file_path: str
    testing_file_path: str
    feature_store_file_path: str