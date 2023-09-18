from zenml import pipeline


from steps.ingest_data import ingest_df
from steps.data_cleaning import cleaning_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model
import logging

#Define a ZenML pipeline called training_pipeline.
@pipeline(enable_cache=False)
def train_pipeline(data_path:str):
    '''
    Data pipeline for training the model.

    Args:
        data_path (str): The path to the data to be ingested.
    '''
    #step ingesting data: returns the data.
    df = ingest_df(data_path=data_path)
    #step to clean the data.
    X_train, X_test, y_train, y_test = cleaning_data(df=df)
    #training the model
    model = train_model(X_train=X_train,y_train=y_train)
    #Evaluation metrics of data
    confusion_matrix, classification_report, accuracy_score, precision_score, recall_score = evaluate_model(model=model,X_test=X_test, y_test=y_test)  
    