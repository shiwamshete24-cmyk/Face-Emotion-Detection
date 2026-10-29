
import cv2
import numpy as np

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from tensorflow.keras.models import model_from_json

BASE_DIR = Path(__file__).resolve().parent

LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

model = None
face_cascade = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, face_cascade

    json_path = BASE_DIR / "emotiondetector.json"
    weights_path = BASE_DIR / "emotiondetector.h5"
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    with open(json_path, "r", encoding="utf-8") as file:
        model = model_from_json(file.read())

    model.load_weights(str(weights_path))
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError("Could not load Haar cascade.")

    print("Emotion detection model loaded successfully!")
    yield


app = FastAPI(
    title="Face Emotion Detection API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this before production deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or face_cascade is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready."
        )

    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG or WebP image."
        )

    contents = await file.read()

    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 5 MB."
        )

    image_array = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image."
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    results = []

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype(np.float32) / 255.0
        face = face.reshape(1, 48, 48, 1)

        predictions = model.predict(face, verbose=0)[0]
        index = int(np.argmax(predictions))

        results.append({
            "emotion": LABELS[index],
            "confidence": round(float(predictions[index]) * 100, 2),
            "box": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            }
        })

    return {
        "faces_detected": len(results),
        "predictions": results
    }
