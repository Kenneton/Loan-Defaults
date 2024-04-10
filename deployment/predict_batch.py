import requests
import pandas as pd
import json

data = pd.read_pickle("test.pkl")

sample = data.iloc[0:50]
payload = sample.to_json(orient="records")  

url = "https://loan-application-service-zem3i4ajwq-lz.a.run.app/predict_batch" 

response = requests.post(url, json=json.loads(payload))

if response.status_code == 200:
    print("Response from the model:", response.json())
else:
    print("Failed to get a response, status code:", response.status_code)
