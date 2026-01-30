import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model/image_classifier.pth")
DATA_DIR = "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# LOAD CLASS NAMES
# ---------------------------
class_names = sorted(os.listdir(os.path.join(DATA_DIR, "train")))

# ---------------------------
# LOAD MODEL
# ---------------------------
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---------------------------
# IMAGE TRANSFORM
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------------
# FASTAPI APP
# ---------------------------
app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid image file")

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    return JSONResponse(
        content={
            "predicted_class": class_names[predicted_idx.item()],
            "confidence": round(confidence.item(), 4)
        }
    )
