import logging

import pandas as pd 
from src.clean_data import DataPreprocessing, LabelEncoding


# Function to get data for testing purposes
def get_data_for_test():
    try:
        df = pd.read_csv('./data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        df = df.sample(n=100)
        data_preprocessing = DataPreprocessing()
        data = data_preprocessing.handle_data(df)  
        
        # Instantiate the FeatureEncoding strategy
        label_encode = LabelEncoding()
        df_encoded = label_encode.handle_data(data) 
        df_encoded.drop(['Churn'],axis=1,inplace=True)
        
        result = df_encoded.to_json(orient="split")
        return result
    except Exception as e:
        logging.error("e")
        raise e