import io
import os

import torch
import wandb
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from omegaconf import OmegaConf

from src.lightning_module import XrayClassifier
from src.factories import build_transforms

WANDB_ARTIFACT = os.getenv(
    "WANDB_ARTIFACT",
    "mustaphamensouri/lung-pathology-multilabel/lung-pathology-classifier:production",
)

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Chest X-ray Multi-label Classifier API")

model = None
val_tf = None
labels = None


@app.on_event("startup")
def load_model():
    global model, val_tf, labels

    api = wandb.Api()
    artifact = api.artifact(WANDB_ARTIFACT)
    artifact_dir = artifact.download()

    ckpt_path = os.path.join(artifact_dir, "model.ckpt")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    hparams = ckpt["hyper_parameters"]
    labels = hparams["class_names"]
    if labels is None:
        raise RuntimeError("Checkpoint does not contain class_names. Retrain or repackage the model.")
    cfg = hparams["cfg"]
    
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    num_classes = hparams.get("num_classes", len(labels))
    max_epochs = hparams.get("max_epochs", 1)

    model = XrayClassifier(
        cfg=cfg,
        num_classes=num_classes,
        class_names=labels,
        max_epochs=max_epochs,
    )

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)

    _, val_tf = build_transforms(cfg.augmentation)

    print(f"Model loaded successfully from {WANDB_ARTIFACT}")
    print(f"Device: {device}")
    print(f"Number of labels: {len(labels)}")


@app.get("/")
def root():
    return {"message": "Chest X-ray classifier API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "loading",
        "device": device,
        "num_labels": len(labels) if labels else None,
        "artifact": WANDB_ARTIFACT,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 10):
    if model is None or val_tf is None or labels is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    x = val_tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    results = [
        {"label": label, "probability": float(prob)}
        for label, prob in zip(labels, probs)
    ]

    results = sorted(results, key=lambda x: x["probability"], reverse=True)

    return {
        "filename": file.filename,
        "top_k": top_k,
        "predictions": results[:top_k],
        "disclaimer": "Research/demo use only. Not for medical diagnosis.",
    }