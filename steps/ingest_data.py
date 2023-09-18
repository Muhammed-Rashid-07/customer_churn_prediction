import pandas as pd
import numpy as np
import logging
from zenml import step



class IngestData:
    """
    Ingesting data to the workflow.
    """
    def __init__(self, path:str) -> None:
        """
        Args:
            data_path(str): path of the datafile 
        """
        self.path = path
    
    def get_data(self):
        df = pd.read_csv(self.path)
        logging.info("Reading csv file successfully completed.")
        return df
    

@step(enable_cache = False)
def ingest_df(data_path:str) -> pd.DataFrame:
    """
    ZenML step for ingesting data from a CSV file.
    
    Args:
        data_path (str): Path of the datafile to be ingested.
    
    Returns:
        df (pd.DataFrame): DataFrame containing the ingested data.
    """
    try:
        #Creating an instance of IngestData class and ingest the data
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        logging.info("Ingesting data completed")
        return df
    except Exception as e:
        #Log an error message if data ingestion fails and raise the exception
        logging.error("Error while ingesting data")
        raise e
    