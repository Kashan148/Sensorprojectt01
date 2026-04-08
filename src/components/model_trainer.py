import sys
from typing import Generator, List, Tuple
import os
import numpy as np  
import pandas as pd
from sklearn.metrics import accuracy_score


from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils


from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join(artifact_folder)
    trained_model_file_path = os.path.join(artifact_folder, 'model.pkl')
    expected_accuracy = 0.45
    model_config_file_path = os.path.join('config', 'model.yaml')

    