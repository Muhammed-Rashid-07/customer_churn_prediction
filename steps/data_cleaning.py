import pandas as pd
import numpy as np
from src.clean_data import DataPreprocessing,DataDivideStrategy,FeatureEncoding
import logging
from typing_extensions import Annotated
from typing import Tuple
from zenml import step



# Define a ZenML step for cleaning and preprocessing data
@step(enable_cache = False)
def cleaning_data(df:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"],
]:
    try:
        # Instantiate the DataPreprocessing strategy
        data_preprocessing = DataPreprocessing()
        data = data_preprocessing.handle_data(df)  
        
        # Instantiate the FeatureEncoding strategy
        feature_encode = FeatureEncoding()
        df_encoded = feature_encode.handle_data(data)  

        # Instantiate the DataDivideStrategy strategy
        split_data = DataDivideStrategy()
        X_train, X_test, y_train, y_test = split_data.handle_data(df_encoded)
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.error("Error in step cleaning data",e)
        raise e
        
        