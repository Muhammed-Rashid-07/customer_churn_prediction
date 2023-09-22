import json
import logging
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main


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

    st.markdown(   """ 
    #### Description of Features 
    This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
            | Feature Name                  | Description                                                                               |
            | -----------------------------        | -----------------------------------------------                                    |
            | SeniorCitizen                        | Indicates whether the customer is a senior citizen.                                |
            | tenure                               | Number of months the customer has been with the company.                           |
            | MonthlyCharges                       | Monthly charges incurred by the customer.                                          |
            | TotalCharges                         | Total charges incurred by the customer.                                            |
            | gender_Male                          | Gender of the customer (Male: 1, Female: 0).                                       |
            | Partner_Yes                          | Whether the customer has a partner (Yes: 1, No: 0).                                |
            | Dependents_Yes                       | Whether the customer has dependents (Yes: 1, No: 0).                               |
            | PhoneService_Yes                     | Whether the customer has a phone service (Yes: 1, No: 0).                          |
            | MultipleLines_Yes                    | Whether the customer has multiple lines (Yes: 1, No: 0).                           |
            | InternetService_Fiber optic          | Type of internet service (Fiber optic: 1, No: 0).                                  |
            | InternetService_No                   | Type of internet service (No: 1, Other: 0).                                        |
            | OnlineSecurity_Yes                   | Whether the customer has online security service (Yes: 1, No: 0).                  |
            | OnlineBackup_Yes                     | Whether the customer has online backup service (Yes: 1, No: 0).                    |
            | DeviceProtection_Yes                 | Whether the customer has device protection service (Yes: 1, No: 0).                |
            | TechSupport_Yes                      | Whether the customer has tech support service (Yes: 1, No: 0).                     |
            | StreamingTV_Yes                      | Whether the customer has streaming TV service (Yes: 1, No: 0).                     |
            | StreamingMovies_Yes                  | Whether the customer has streaming movies service (Yes: 1, No: 0).                 |
            | Contract_One year                    | Type of contract (One year: 1, Other: 0).                                          |
            | Contract_Two year                    | Type of contract (Two year: 1, Other: 0).                                          |
            | PaperlessBilling_Yes                 | Whether the customer has paperless billing (Yes: 1, No: 0).                        |
            | PaymentMethod_Credit card (automatic)| Payment method (Credit card: 1, Other: 0).                                         |
            | PaymentMethod_Electronic check       | Payment method (Electronic check: 1, Other: 0).                                    |
            | PaymentMethod_Mailed check           | Payment method (Mailed check: 1, Other: 0).                                        |
            | Churn_Yes                            | Whether the customer has churned (Yes: 1, No: 0).                                  |
            
             
         

            """)
    # Define the data columns with their respective values
    SeniorCitizen = st.selectbox("Are you senior citizen?",
            options=[True, False],)
    tenure = st.number_input("Tenure")
    MonthlyCharges = st.number_input("Monthly Charges: ")
    TotalCharges = st.number_input("Total Charges: ")
    gender_Male = st.radio("Male", options=[True, False])
    Partner_Yes = st.radio("Do you have a partner? ", options=[True, False])
    Dependents_Yes = st.radio("Dependents: ", options=[True, False])
    PhoneService_Yes = st.radio("Do you have phone service? : ", options=[True, False])
    MultipleLines_Yes = st.radio("Do you Multiplines? ", options=[True, False])
    InternetService_Fiber_optic = st.radio("Did you subscribe to Internet Service Fibre optic? ", options=[True, False])
    InternetService_No = st.radio("Did you subscribe for Internet service? ", options=[True, False])
    OnlineSecurity_Yes = st.radio("Did you subscribe for OnlineSecurity? ", options=[True, False])
    OnlineBackup_Yes = st.radio("Did you subscribe for Online Backup service? ", options=[True, False])
    DeviceProtection_Yes = st.radio("Did you subscribe for device protection only?", options=[True, False])
    TechSupport_Yes =st.radio("Did you subscribe for tech support? ", options=[True, False])
    StreamingTV_Yes = st.radio("Did you subscribe for TV streaming", options=[True, False])
    StreamingMovies_Yes = st.radio("Did you subscribe for streaming movies? ", options=[True, False])
    Contract_One_year = st.radio("Did you have 1 year contract?", options=[True, False])
    Contract_Two_year = st.radio("Did you have 2 year contracts? ", options=[True, False])
    PaperlessBilling_Yes = st.radio("Do you use paperless billing? ", options=[True, False])
    PaymentMethod_Credit_card_automatic = st.radio("Do you use credit card as payment method? ", options=[True, False])
    PaymentMethod_Electronic_check = st.radio("Do you use electronic payments? ", options=[True, False])
    PaymentMethod_Mailed_check = st.radio("Do you use mailed check for payments? ", options=[True, False])

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
            'gender_Male': int(gender_Male),
            'Partner_Yes': int(Partner_Yes),
            'Dependents_Yes': int(Dependents_Yes),
            'PhoneService_Yes': int(PhoneService_Yes),
            'MultipleLines_Yes': int(MultipleLines_Yes),
            'InternetService_Fiber optic': int(InternetService_Fiber_optic), 
            'InternetService_No': int(InternetService_No),
            'OnlineSecurity_Yes': int(OnlineSecurity_Yes),
            'OnlineBackup_Yes': int(OnlineBackup_Yes),
            'DeviceProtection_Yes': int(DeviceProtection_Yes),
            'TechSupport_Yes': int(TechSupport_Yes),
            'StreamingTV_Yes': int(StreamingTV_Yes),
            'StreamingMovies_Yes': int(StreamingMovies_Yes),
            'Contract_One year': int(Contract_One_year), 
            'Contract_Two year': int(Contract_Two_year), 
            'PaperlessBilling_Yes': int(PaperlessBilling_Yes),
            'PaymentMethod_Credit card (automatic)': int(PaymentMethod_Credit_card_automatic),
            'PaymentMethod_Electronic check': int(PaymentMethod_Electronic_check), 
            'PaymentMethod_Mailed check': int(PaymentMethod_Mailed_check)
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