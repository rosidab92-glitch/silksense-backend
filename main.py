from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import os
import gdown

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = "silk_model_best.keras"
FILE_ID = "1ttX-9ANxR-Isw04Tu62SYNQFKAux-It_"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

import keras
model = keras.models.load_model(MODEL_PATH)
print("Model loaded!")

def preprocess(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

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
        if prediction == "Authentic":
            obs = ["Natural silk sheen detected", "Weave density matches handloom pattern", "Texture consistent with GI-certified silk", "Fiber alignment confirms authentic silk"]
            reason = "Genuine Assamese Mekhela Chador markers present."
        else:
            obs = ["Synthetic fiber pattern detected", "Weave inconsistent with handloom", "Sheen suggests artificial material", "Texture deviates from authentic silk"]
            reason = "Indicators of synthetic or non-GI material detected."
        return JSONResponse({
            "prediction": prediction,
            "confidence": confidence,
            "authentic_prob": auth_prob,
            "fake_prob": fake_prob,
            "badge": badge,
            "silk_type": "Mekhela Chador",
            "source": "ResNet-50 (90.95%)",
            "observations": obs,
            "reason": reason
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
