import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title = 'Titanic Survival API')

model = joblib.load('titanic_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

class PassengerInput(BaseModel):
    Sex: int
    Age: int
    SibSp: int
    Parch: int
    Embarked_C: int
    Embarked_Q: int
    Embarked_S: int
    Title_Dr: int
    Title_Mr: int
    Title_Mrs: int
    Title_Don: int
    Title_Col: int
    Title_Capt: int
    Title_Jonkheer: int
    Title_the_Countess: int
    Title_Major: int
    Title_Rev: int
    Fare: float|int
    Pclass_1: int
    Pclass_2: int
    Pclass_3: int

@app.post('/predict')
def predict_survival(data: PassengerInput):
    df = pd.DataFrame([data.model_dump()])
    df = df.reindex(columns = model_columns , fill_value = 0)
    df = scaler.transform(df)

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return {'Prediction': prediction, 'Probability': probability}