import pandas as pd 
from fastapi import FastAPI
import joblib
from inputs import DynamicModel
from typing import List

threshold = 0.78

model = joblib.load("final_model.joblib")
dtypes = pd.read_pickle("dtypes.pkl")

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(data: DynamicModel):

    X = pd.DataFrame([data.dict()]).astype(dtypes)

    y_proba = model.predict_proba(X)[:, 1]
    y = (y_proba >= threshold).astype(int)

    result = {"result": int(y[0]), "probability": y_proba[0]}

    return result


@app.post("/predict_batch")
async def predict_batch(data: List[DynamicModel]):
    X_batch = pd.DataFrame([item.dict() for item in data])
    X_batch = X_batch.astype(dtypes)

    y_proba = model.predict_proba(X_batch)[:, 1]
    y = (y_proba >= threshold).astype(int)

    results = y.tolist()
    probabilities = y_proba.tolist()
                       
    return {"results": results, "probability": probabilities}

