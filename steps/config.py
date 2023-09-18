from zenml.steps import BaseParameters


"""
This file is used for used for configuring
and specifying various parameters related to 
your machine learning models and training process
"""

class ModelName(BaseParameters):
    """
    Model configurations
    """
    model_name: str = "logistic regression"
     
    