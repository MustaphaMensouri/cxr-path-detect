import io
import torch
import wandb
from PIL import Image
from fastapi import FastAPI, UploadFile, File

from src.lightning_module import XrayClassifier
from src.factories import build_transforms
from src.datamodule import load_labels

CONFIG_PATH = "configs/config.yaml"
LABELS_PATH = "../data_marts/lung_PA_AP_AP_horizontal/labels_used.txt"
WANDB_ARTIFACT = "mustaphamensouri/lung-pathology-multilabel/model-2loyqw63:v0"

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Chest X-ray Multi-label Classifier API")

model = None
val_tf = None
labels = None


@app.on_event("startup")
def load_model():
    global model, val_tf, labels

    labels = load_labels(LABELS_PATH)

    api = wandb.Api()
    artifact = api.artifact(WANDB_ARTIFACT)
    artifact_dir = artifact.download()

    ckpt_path = f"{artifact_dir}/model.ckpt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    hparams = ckpt["hyper_parameters"]
    cfg = hparams["cfg"]
    num_classes = hparams["num_classes"]
    max_epochs = hparams["max_epochs"]

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

    print("Model loaded successfully")


@app.get("/")
def root():
    return {"message": "Chest X-ray classifier API is running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "num_labels": len(labels) if labels else None,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 10):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

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