from fastapi import FastAPI
from pydantic import BaseModel 
import joblib

#initialize the app

app = FastAPI (title = "Spam Detection", description="This is a simple spam detection API", version="0.1")

#load the model

model = joblib.load("spam_classifier_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

#defining the request body

class Message(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "Welcome to the Spam Detection API!"}

@app.post("/predict")
def predict_spam(data: Message):
    # Transform the input text
    text_vectorized = vectorizer.transform([message.text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)
    
    # Return response
    return {"text": Message.text, "prediction": "Spam" if prediction[0] == 1 else "Ham"}
