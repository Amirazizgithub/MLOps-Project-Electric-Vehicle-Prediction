# Path: Electric_Vehicle_Prediction/components/data_validation.py

import json
import sys

import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

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