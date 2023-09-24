import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import train_test_split
from abc import abstractmethod, ABC
from typing import Union
from sklearn.preprocessing import LabelEncoder

# Define an abstract class which carries the blueprint of other strategies
class DataStrategy(ABC):
    """
    Abstract class defining for handling data.
    """
    @abstractmethod
    def handle_data(self, df:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        """
        Abstract method for handling data.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Union[pd.DataFrame, pd.Series]: Processed DataFrame or Series.
        """
        pass
    
    
# Data Preprocessing strategy
class DataPreprocessing(DataStrategy):
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Preprocesses the input DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Union[pd.DataFrame, pd.Series]: Preprocessed DataFrame.
        """
        try:
            # Filling empty columns with 0
            df['TotalCharges'] = df['TotalCharges'].replace(' ', 0)

            # Changing the total charges from string into float
            df['TotalCharges'] = df['TotalCharges'].astype(float)
            
            df.drop('customerID', axis=1, inplace=True)
            
            df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0}).astype(int)
            
            # Services provided
            service = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
            for col in service:
                df[col] = df[col].replace({'No phone service': 'No', 'No internet service': 'No'})

            return df
        except Exception as e:
            logging.error("Error in Preprocessing", e)
            raise e
    
    
# Feature Encoding Strategy
class LabelEncoding(DataStrategy):
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Encodes categorical features using Label Encoding.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Union[pd.DataFrame, pd.Series]: DataFrame with encoded categorical features.
        """
        try:
            df_cat = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                      'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                      'TechSupport', 'StreamingTV',  'StreamingMovies', 'Contract', 
                      'PaperlessBilling', 'PaymentMethod']
            lencod = LabelEncoder()

            for col in df_cat:
                df[col] = lencod.fit_transform(df[col])
                
            return df
        except Exception as e:
            logging.error(e)
            raise e
    
# Data splitting Strategy
class DataDivideStrategy(DataStrategy):
    def handle_data(self, df:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Splits the input DataFrame into training and testing sets.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Union[pd.DataFrame, pd.Series]: Training and testing sets for features and labels.
        """
        try:
            X = df.drop('Churn', axis=1)
            y = df['Churn']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in DataDividing", e)
            raise e
