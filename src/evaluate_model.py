import logging
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, f1_score, recall_score
from abc import ABC,abstractmethod
import numpy as np


# Abstract class for model evaluation
class Evaluate(ABC):
    """
        Abstract method to evaluate a machine learning model's performance.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Evaluation result.
    """
    @abstractmethod
    def evaluate_model(self, y_true:np.ndarray, y_pred:np.ndarray):
        pass
    
    
#Class to calculate accuracy score 
class Accuracy_score(Evaluate):
    """
        Calculates and returns the accuracy score for a model's predictions.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Accuracy score as a percentage.
    """
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            accuracy_scr = accuracy_score(y_true=y_true, y_pred=y_pred) * 100
            logging.info("Accuracy_score:", accuracy_scr)  # Log accuracy score as a float
            return accuracy_scr  # Return the accuracy score as a float
        except Exception as e:
            logging.error("Error in evaluating the accuracy of the model",e)
            raise e


#Class to evaluate confusion matrix
class ConfusionMatrix(Evaluate):
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates and returns the confusion matrix for a model's predictions.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: Confusion matrix.
        """
        try:
            confusion_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
            logging.info("Confusion Matrix: ",confusion_mat)
            return confusion_mat
        except Exception as e:
            logging.error("Error in evaluating the confusion matrix", e)
            raise e
        
        
class ClassificationReport(Evaluate):
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Generates and returns a classification report for a model's predictions.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            str: Classification report as a string.
        """
        try:
            classification_rpt = classification_report(y_pred = y_pred, y_true = y_true)
            logging.info("Classification report: ",classification_report)
            return classification_rpt
        except Exception as e:
            logging.error("Error in evaluating classification report:",e)
            raise e
        
class Precision_Score(Evaluate):
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Generates and returns a precision score for a model's predictions.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            precision_score: float
        """
        try:
            precision = precision_score(y_true=y_true,y_pred=y_pred)
            logging.info("Precision score: ",precision)
            return float(precision)
        except Exception as e:
            logging.error("Error in calculation of precision_score",e)
            raise e
        
class F1_Score(Evaluate):
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Generates and returns a F1 score for a model's predictions.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            f1_score: float
        """
        try:
            f1_scr = f1_score(y_pred=y_pred,y_true=y_true)
            logging("F1 score: ",f1_scr)
            return float(f1_scr)
        except Exception as e:
            logging.info("Error in calculating F1 score",e)
            raise e            
        
        
class Recall_Score(Evaluate):
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Generates and returns a recall score for a model's predictions.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            recall_score: float
        """
        try:
            recall = recall_score(y_pred=y_pred, y_true=y_true)
            logging.info("Recall_Score: ",recall)
            return float(recall)
        except Exception as e:
            logging.error("Error in calculating Recall score",e)
            raise e