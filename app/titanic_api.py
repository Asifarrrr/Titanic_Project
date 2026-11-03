import joblib
import pandas as pd
from fastapi import FastAPI,Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title = 'Titanic Survival API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/data")
def data():
    df = joblib.load(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Titanic_Project\models\data_frame.pkl")
    return Response(df.to_json(orient="records"), media_type="application/json")


model = joblib.load(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Titanic_Project\data\titanic_model.pkl")
scaler = joblib.load(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Titanic_Project\models\scaler.pkl")
model_columns = joblib.load(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Titanic_Project\models\model_columns.pkl")

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