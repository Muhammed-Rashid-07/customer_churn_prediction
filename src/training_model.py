import pandas as pd
from sklearn.linear_model import LogisticRegression
from abc import ABC, abstractmethod
import logging
from sklearn.preprocessing import StandardScaler



#Abstract model
class Model(ABC):
    @abstractmethod
    def train(self,X_train:pd.DataFrame,y_train:pd.Series):
        """
        Trains the model on given data
        """
        pass
    

class LogisticReg(Model):
    """
    Implementing the Logistic Regression model.
    """
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Training the model
        
        Args:
            X_train: pd.DataFrame,
            y_train: pd.Series
        """
        logistic_reg = LogisticRegression()
        logistic_reg.fit(X_train,y_train)
        return logistic_reg
    