import logging
import pandas as pd
import numpy as np
import mlflow
from zenml import step
from src.evaluate_model import ClassificationReport, ConfusionMatrix, Accuracy_score, Recall_Score, F1_Score, Precision_Score
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import ClassifierMixin
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name, enable_cache = False)
def evaluate_model(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[
    Annotated[np.ndarray,"confusion_matix"],
    Annotated[str,"classification_report"],
    Annotated[float,"accuracy_score"],
    Annotated[float,"precision_score"],
    Annotated[float,"recall_score"]
    ]:
    """
    Evaluate a machine learning model's performance using common metrics.

    Args:
        model (ClassifierMixin): The trained classification model.
        X_test (pd.DataFrame): Independent test data.
        y_test (pd.Series): True labels for the test data.

    Returns:
        Tuple:
            - confusion_matrix (np.ndarray): A confusion matrix.
            - classification_report (str): A text report with classification metrics.
            - accuracy_score (tuple): A tuple containing accuracy score as a float and '%' symbol.
    """
    try:
        y_pred =  model.predict(X_test)
        
        confusion_mx_class = ConfusionMatrix()
        confusion_matrix = confusion_mx_class.evaluate_model(y_true=y_test, y_pred=y_pred)
       
        classification_report_class = ClassificationReport()
        classification_report = classification_report_class.evaluate_model(y_true=y_test, y_pred=y_pred)
        
        recall_score_class = Recall_Score()
        recall_score = recall_score_class.evaluate_model(y_pred=y_pred, y_true=y_test)
        mlflow.log_metric("Recall_score ",recall_score)
        
        precision_score_class = Precision_Score()
        precision_score = precision_score_class.evaluate_model(y_pred=y_pred,y_true=y_test)
        mlflow.log_metric("Precision_score ",precision_score)
        
        accuracy_score_class = Accuracy_score()
        accuracy_score = accuracy_score_class.evaluate_model(y_true=y_test, y_pred=y_pred)
        
        
        
        logging.info("accuracy_score:",accuracy_score)
    
        return confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
    
    except Exception as e:
        logging.error("Error in evaluating model",e)
        raise e    