from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import uvicorn
from typing import List
import os

app = FastAPI(
    title="Music Genre Recommender API",
    description="An API that recommends music genres based on user age and gender",
    version="1.0.0"
)

# Path to model file
MODEL_PATH = "music_recommender.joblib"

# Load the model or handle missing model gracefully
try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    print(f"Warning: Model file '{MODEL_PATH}' not found. Prediction endpoint will return an error.")

class UserInput(BaseModel):
    age: int = Field(..., description="User's age", ge=1, le=120)
    gender: int = Field(..., description="User's gender (0 for female, 1 for male)", ge=0, le=1)

class PredictionResponse(BaseModel):
    genre: str
    confidence: float = None

@app.get("/")
def read_root():
    return {
        "message": "Music Genre Recommender API",
        "endpoints": {
            "/predict": "POST endpoint to predict music genre",
            "/docs": "API documentation",
            "/health": "API health check"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(user_input: UserInput):
    if not model_loaded:
        raise HTTPException(status_code=503, detail=f"Model file '{MODEL_PATH}' not found. Service unavailable.")
    
    try:
        # Make prediction
        prediction = model.predict([[user_input.age, user_input.gender]])
        
        # Get prediction probability if the model supports it
        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([[user_input.age, user_input.gender]])
            # Get the highest probability
            confidence = float(max(probabilities[0]))
        
        return {
            "genre": prediction[0],
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)