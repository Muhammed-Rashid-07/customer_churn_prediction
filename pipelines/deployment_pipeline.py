import numpy as np
import logging
import pandas as pd
from materializer.custom_materializer import cs_materializer
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from materializer.custom_materialzier import cs_materializer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output


from steps.data_cleaning import cleaning_data
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_data
from steps.model_train import train_model


# Define Docker settings with MLflow integration
docker_settings = DockerSettings(required_integrations = {MLFLOW})


#Define class for deployment pipeline configuration
class DeploymentTriggerComfig(BaseParameters):
    def __init__(self):
        self.min_accuracy:float = 0.50
    
    
#Define step which trigger the deployment only the accuracy score met the  threshold score.
@step 
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerComfig,
):
    """
    It trigger the deployment only if accuracy is greater than min accuracy.
    Args:
        accuracy: accuracy of the model.
        config: Minimum accuracy thereshold.
    """
    try:
        return accuracy >= config.min_accuracy
    except Exception as e:
        logging.error("Error in deployment trigger",e)
        raise e
    


@pipeline(enable_cache=True,setting={"docker_settings":docker_settings})
def continuous_deployment_pipeline(
    min_accuracy:float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    # Step 1: Ingesting data
    # Ingest data from a specified data path
    df = ingest_data()

    # Step 2: Data Cleaning
    # Clean the ingested data
    X_train, X_test, y_train, y_test = cleaning_data(df=df)

    # Step 3: Model Training
    # Train a machine learning model using the cleaned data
    model = train_model(X_train=X_train, y_train=y_train)

    # Step 4: Model Evaluation
    # Evaluate the trained model's performance
    confusion_matrix, classification_report, accuracy_score, precision_score, recall_score = evaluate_model(model=model, X_test=X_test, y_test=y_test)
    deployment_decision = DeploymentTriggerComfig(accuracy_score)
    mlflow_model_deployer_step(
        model=model,
        deployment_decision = deployment_decision,
        workers = workers,
        timeout = timeout
    )