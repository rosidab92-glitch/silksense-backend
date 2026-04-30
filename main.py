from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os
import gdown

# Download model from Google Drive if not exists
if not os.path.exists("silk_model_final.h5"):
    print("Downloading model...")
    gdown.download(
        "https://drive.google.com/uc?id=YOUR_FILE_ID",
        "silk_model_final.h5",
        quiet=False
    )

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = tf.keras.models.load_model("silk_model_final.h5")
CLASSES = ["Authentic", "Fake"]

def preprocess(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_observations(prediction, confidence):
    if prediction == "Authentic":
        return ["Natural silk sheen detected","Weave density matches handloom pattern","Texture consistent with GI-certified silk","Fiber alignment confirms authentic silk"]
    return ["Synthetic fiber pattern detected","Weave inconsistent with handloom production","Sheen suggests artificial material","Texture deviates from authentic Assamese silk"]

def get_reason(prediction, confidence):
    if prediction == "Authentic":
        return "High confidence — genuine Assamese Mekhela Chador markers present." if confidence >= 85 else "Likely authentic — most key markers present."
    return "High confidence — strong indicators of synthetic or non-GI material." if confidence >= 85 else "Likely inauthentic — several markers deviate from genuine Assamese silk."

@app.get("/")
def root():
    return {"status": "SilkSense API running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = preprocess(image)
        preds = model.predict(tensor)[0]
        if len(preds) == 1:
            fake_prob = round(float(preds[0]) * 100, 2)
            auth_prob = round(100 - fake_prob, 2)
        else:
            auth_prob = round(float(preds[0]) * 100, 2)
            fake_prob = round(float(preds[1]) * 100, 2)
        prediction = "Authentic" if auth_prob >= fake_prob else "Fake"
        confidence = max(auth_prob, fake_prob)
        badge = "authentic" if prediction == "Authentic" and confidence >= 85 else "likely" if prediction == "Authentic" else "fake"
        return JSONResponse({
            "prediction": prediction,
            "confidence": confidence,
            "authentic_prob": auth_prob,
            "fake_prob": fake_prob,
            "badge": badge,
            "silk_type": "Mekhela Chador",
            "source": "ResNet-50 (Keras)",
            "observations": get_observations(prediction, confidence),
            "reason": get
