from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Fraud Detection API",
    description="Machine Learning API for detecting fraudulent transactions",
    version="1.0"
)

# Load model
model = joblib.load("models/model.pkl")

# Dataset columns
columns = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount"
]

# Request schema
class TransactionData(BaseModel):
    data: list

@app.get("/")
def home():
    return {
        "message": "Fraud Detection API Running"
    }

@app.post("/predict")
def predict(transaction: TransactionData):

    # Convert input into dataframe
    input_df = pd.DataFrame(
        [transaction.data],
        columns=columns
    )

    prediction = model.predict(input_df)

    result = int(prediction[0])

    if result == 1:
        message = "Fraudulent Transaction Detected"
    else:
        message = "Normal Transaction"

    return {
        "prediction": result,
        "message": message
    }