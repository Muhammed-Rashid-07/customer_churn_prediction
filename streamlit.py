import json
import logging
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import run_main


def main():
    st.title("End to End Customer Satisfaction Pipeline with ZenML")

   
    st.markdown(
        """ 
    #### Problem Statement 
     The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.    """
    )
   
    st.markdown(
        """ 
    Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    """
    )
    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
    | Models        | Description   | 
    | ------------- | -     | 
    | SeniorCitizen | Indicates whether the customer is a senior citizen. | 
    | tenure   | Number of months the customer has been with the company. |  
    | MonthlyCharges  |  Monthly charges incurred by the customer. | 
    | TotalCharges | Total charges incurred by the customer. |
    | gender | Gender of the customer (Male: 1, Female: 0). | 
    | Partner | Whether the customer has a partner (Yes: 1, No: 0). |
    | Dependents |  Whether the customer has dependents (Yes: 1, No: 0). |
    | PhoneService  | Whether the customer has dependents (Yes: 1, No: 0). |   
    | MultipleLines | Whether the customer has multiple lines (Yes: 1, No: 0). | 
    | InternetService | Type of internet service (No: 1, Other: 0). | 
    | OnlineSecurity | Whether the customer has online security service (Yes: 1, No: 0). | 
    | OnlineBackup | Whether the customer has online backup service (Yes: 1, No: 0). | 
    | DeviceProtection | Whether the customer has device protection service (Yes: 1, No: 0). | 
    | TechSupport  | Whether the customer has tech support service (Yes: 1, No: 0). |
    | StreamingTV  | Whether the customer has streaming TV service (Yes: 1, No: 0). |
    | StreamingMovies  | Whether the customer has streaming movies service (Yes: 1, No: 0). |
    | Contract | Type of contract (One year: 1, Other: 0). |
    | PaperlessBilling | Whether the customer has paperless billing (Yes: 1, No: 0). |
    | PaymentMethod  | Payment method (Credit card: 1, Other: 0). |
    | Churn   | Whether the customer has churned (Yes: 1, No: 0).   |
    
    """
    )
    

    payment_options = {
    2: "Electronic check",
    3: "Mailed check",
    1: "Bank transfer (automatic)",
    0: "Credit card (automatic)"
    }
    
    contract = {
        0: "Month-to-month",
        2: "Two year",
        1: "One year"
    }
    
    def format_func(PaymentMethod):
        return payment_options[PaymentMethod]
    
    
    def format_func_contract(Contract):
        return contract[Contract]
    
    display = ("male", "female")
    options = list(range(len(display)))
    # Define the data columns with their respective values
    SeniorCitizen = st.selectbox("Are you senior citizen?",
            options=[True, False],)
    tenure = st.number_input("Tenure")
    MonthlyCharges = st.number_input("Monthly Charges: ")
    TotalCharges = st.number_input("Total Charges: ")
    gender = st.radio("gender:", options, format_func=lambda x: display[x])
    Partner = st.radio("Do you have a partner? ", options=[True, False])
    Dependents = st.radio("Dependents: ", options=[True, False])
    PhoneService = st.radio("Do you have phone service? : ", options=[True, False])
    MultipleLines = st.radio("Do you Multiplines? ", options=[True, False])
    InternetService = st.radio("Did you subscribe for Internet service? ", options=[True, False])
    OnlineSecurity = st.radio("Did you subscribe for OnlineSecurity? ", options=[True, False])
    OnlineBackup = st.radio("Did you subscribe for Online Backup service? ", options=[True, False])
    DeviceProtection = st.radio("Did you subscribe for device protection only?", options=[True, False])
    TechSupport =st.radio("Did you subscribe for tech support? ", options=[True, False])
    StreamingTV = st.radio("Did you subscribe for TV streaming", options=[True, False])
    StreamingMovies = st.radio("Did you subscribe for streaming movies? ", options=[True, False])
    Contract = st.radio("Duration of contract: ", options=list(contract.keys()), format_func=format_func_contract)
    PaperlessBilling = st.radio("Do you use paperless billing? ", options=[True, False])
    PaymentMethod = st.selectbox("Payment method:", options=list(payment_options.keys()), format_func=format_func)
    # You can use PaymentMethod to get the selected payment method's numeric value


    if st.button("Predict"):
        service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run_main()
        try:
            data_point = {
            'SeniorCitizen': int(SeniorCitizen),
            'tenure': tenure, 
            'MonthlyCharges': MonthlyCharges, 
            'TotalCharges': TotalCharges,
            'gender': int(gender),
            'Partner': int(Partner),
            'Dependents': int(Dependents),
            'PhoneService': int(PhoneService),
            'MultipleLines': int(MultipleLines), 
            'InternetService': int(InternetService),
            'OnlineSecurity': int(OnlineSecurity),
            'OnlineBackup': int(OnlineBackup),
            'DeviceProtection': int(DeviceProtection),
            'TechSupport': int(TechSupport),
            'StreamingTV': int(StreamingTV),
            'StreamingMovies': int(StreamingMovies),
            'Contract': int(Contract), 
            'PaperlessBilling': int(PaperlessBilling),
            'PaymentMethod': int(PaymentMethod)
        }

            # Convert the data point to a Series and then to a DataFrame
            data_point_series = pd.Series(data_point)
            data_point_df = pd.DataFrame(data_point_series).T

            # Convert the DataFrame to a JSON list
            json_list = json.loads(data_point_df.to_json(orient="records"))
            data = np.array(json_list)
            for i in range(len(data)):
                logging.info(data[i])
            pred = service.predict(data)
            logging.info(pred)
            st.success(f"Customer churn prediction: {'Churn' if pred == 1 else 'No Churn'}")
        except Exception as e:
            logging.error(e)
            raise e

        
if __name__ == "__main__":
    main()