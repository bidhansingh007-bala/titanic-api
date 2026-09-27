import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. API App start karein
app = FastAPI(title="Titanic Survival Prediction API")

# 2. Apna saved XGBoost model load karein ('rb' matlab Read Binary)
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# 3. Input Data ka format define karein (Pydantic ka use karke)
# Ye ensure karega ki user galat data (jaise age ki jagah text) na bhej de
class PassengerData(BaseModel):
    pclass: int
    sex: int      # 0 for male, 1 for female
    age: float
    fare: float

# 4. Route/Endpoint banayein jahan POST request aayegi
@app.post("/predict")
def predict_survival(data: PassengerData):
    # API ko mila data DataFrame (Pandas) mein convert karein
    input_df = pd.DataFrame([{
        'pclass': data.pclass,
        'sex': data.sex,
        'age': data.age,
        'fare': data.fare
    }])

    # Model se prediction mangen
    prediction = model.predict(input_df)

    # 0 ko "Did not survive" aur 1 ko "Survived" mein badlein
    if prediction[0] == 1:
        final_result = "Survived"
    else:
        final_result = "Did not survive"

    # Result ko JSON format mein wapas bhejein
    return {
        "status": "success",
        "prediction": final_result
    }