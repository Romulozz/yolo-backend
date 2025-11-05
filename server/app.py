# app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
from ultralytics import YOLO
import io, os, torch

# === Config ===
MODEL_PATH = os.getenv("MODEL_PATH", "models/mi_detector_v1.pt")  # pon aquí tu archivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Si tus clases son personas específicas y quieres renombrarlas:
# CLASS_NAMES = {0: "Persona X", 1: "Persona Y"}
CLASS_NAMES = None  # usa las que trae el modelo (results.names)

# === App ===
app = FastAPI(title="YOLO Face/People API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # en prod, limita a tu dominio del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Carga de modelo (una sola vez) ===
model = YOLO(MODEL_PATH)
if DEVICE == "cuda":
    model.to("cuda")

# Warmup opcional
try:
    model.predict(Image.new("RGB", (640, 640)), conf=0.25, verbose=False)
except Exception:
    pass

# === Schemas ===
class Detection(BaseModel):
    bbox: List[float]   # [x1,y1,x2,y2]
    label: str
    score: float

class PredictResponse(BaseModel):
    detections: List[Detection]

@app.get("/")
def root():
    return {"status": "ok", "device": DEVICE, "model": os.path.basename(MODEL_PATH)}

@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    conf: Optional[float] = Form(0.5),   # el slider del frontend
):
    # Leer imagen como PIL
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Inferencia
    res = model.predict(
        img,
        conf=float(conf) if conf is not None else 0.5,
        imgsz=640,          # ajusta si tu training usó otro tamaño
        verbose=False,
        device=DEVICE,
    )[0]

    dets: List[Detection] = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.detach().cpu().numpy()
        cls  = res.boxes.cls.detach().cpu().numpy()
        sco  = res.boxes.conf.detach().cpu().numpy()

        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, sco):
            c = int(c)
            if CLASS_NAMES is not None:
                label = CLASS_NAMES.get(c, f"class_{c}")
            else:
                # Usa los nombres del modelo entrenado
                label = res.names.get(c, f"class_{c}")
            dets.append(Detection(
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                label=str(label),
                score=float(s),
            ))

    return PredictResponse(detections=dets)
