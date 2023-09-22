import numpy as np
import json
import logging
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output
from src.clean_data import FeatureEncoding
from .utils import get_data_for_test
from steps.data_cleaning import cleaning_data
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

# Define Docker settings with MLflow integration
docker_settings = DockerSettings(required_integrations = {MLFLOW})

########################################## Continuous deployment pipeline ###############################
    
    
#Define class for deployment pipeline configuration
class DeploymentTriggerConfig(BaseParameters):
    min_accuracy:float = 0.92
    
   
#Define step which trigger the deployment only the accuracy score met the  threshold score.
@step 
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
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
    


# Define a continuous pipeline
@pipeline(enable_cache=False,settings={"docker":docker_settings})
def continuous_deployment_pipeline(
    data_path:str,
    min_accuracy:float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    # Step 1: Ingesting data
    # Ingest data from a specified data path
    df = ingest_df(data_path=data_path)

    # Step 2: Data Cleaning
    # Clean the ingested data
    X_train, X_test, y_train, y_test = cleaning_data(df=df)

    # Step 3: Model Training
    # Train a machine learning model using the cleaned data
    model = train_model(X_train=X_train, y_train=y_train)

    # Step 4: Model Evaluation
    # Evaluate the trained model's performance
    confusion_matrix, classification_report, accuracy_score, precision_score, recall_score = evaluate_model(model=model, X_test=X_test, y_test=y_test)
    deployment_decision = deployment_trigger(accuracy=accuracy_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision = deployment_decision,
        workers = workers,
        timeout = timeout
    )
    
        
####################################### Inference pipeline ###################################################


class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """
    MLflow deployment getter parameters
    
    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction server
        step_name: the name of the step that deployed the MLFlow prediction server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model deployed
    """
    pipeline_name: str
    step_name: str
    running: bool = True
    

@step(enable_cache=False)
def dynamic_importer() -> str:
    """
    Dynamically  import the MLFlow prediction server
    return zenml.integrations.mlflow.services.MLFlowDeploymentService
    """
    data = get_data_for_test()
    return data


@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]
 
    
@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df =['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
       'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
       'MultipleLines_Yes', 'InternetService_Fiber optic',
       'InternetService_No', 'OnlineSecurity_Yes', 'OnlineBackup_Yes',
       'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes',
       'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year',
       'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction



@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    prediction = predictor(service=model_deployment_service,data=batch_data)
    return prediction
    