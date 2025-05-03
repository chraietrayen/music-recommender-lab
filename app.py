from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn

app = FastAPI()

# Make sure the path to your model file is correct
model = joblib.load("music_recommender.joblib")

class UserInput(BaseModel):
    age: int
    gender: int

@app.post("/predict")
def predict(user_input: UserInput):
    prediction = model.predict([[user_input.age, user_input.gender]])
    return {"genre": prediction[0]}

if __name__ == "__main__":
    # This is for local development, using Uvicorn to run the app
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
