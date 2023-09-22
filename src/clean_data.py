import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import train_test_split
from abc import abstractmethod,ABC
from typing import Union
from sklearn.linear_model import LogisticRegression


#Define an abstract class which carries the blueprint of other strategies
class DataStrategy(ABC):
    """
    Abstract class defining for handling data.
    """
    @abstractmethod
    def handle_data(self, df:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass
    
    
#Data Preprocessing strategy
class DataPreprocessing(DataStrategy):
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            # Remove rows with missing values and a specific column
            columns_to_replace = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
            for column in columns_to_replace:
                df[column] = df[column].replace('No internet service', 'No')
            df['MultipleLines'] = df['MultipleLines'].replace('No phone service','No')
            df.drop(['customerID'],axis=1,inplace=True)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').astype(float)
            logging.info("length of df: ",len(df.columns))
            df.dropna(inplace=True)
            return df
        except Exception as e:
            logging.error("Error in Preprocessing",e)
            raise e
    
    
# Feature Encoding Strategy
class FeatureEncoding(DataStrategy):
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            # Identify categorical columns and perform one-hot encoding
            categorical_df = df.select_dtypes(include='object').columns.tolist()
            df_encoded = pd.get_dummies(data=df,columns=categorical_df,drop_first=True)
            return df_encoded
        except Exception as e:
            logging.error("Error in Feature Encoding")
            raise e
    
#Data splitting Strategy
class DataDivideStrategy(DataStrategy):
    def handle_data(self, df:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        #Splitting data X,y and then into training and testing
        try:
            X = df.drop('Churn_Yes',axis=1)
            y = df['Churn_Yes']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in DataDividing",e)
            raise e
            
            
            
        
            
    

    

