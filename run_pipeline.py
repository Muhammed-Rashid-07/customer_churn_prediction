from pipelines.training_pipeline import train_pipeline
from zenml.client import Client
if __name__ == '__main__':
    #printimg the experiment tracking uri
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    #Run the pipeline
    train_pipeline(data_path="/mnt/e/Customer_churn/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    