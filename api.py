import io
import os
import json

import torch
import wandb
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from omegaconf import OmegaConf

import base64
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.lightning_module import XrayClassifier
from src.factories import build_transforms

from fastapi.middleware.cors import CORSMiddleware

WANDB_ARTIFACT = os.getenv(
    "WANDB_ARTIFACT",
    "mustaphamensouri/lung-pathology-multilabel/lung-pathology-classifier:v2",
)

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Chest X-ray Multi-label Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
val_tf = None
labels = None
thresholds = None


@app.on_event("startup")
def load_model():
    global model, val_tf, labels,  thresholds

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
    threshold_path = os.path.join(artifact_dir, "thresholds.json")

    if os.path.exists(threshold_path):
        with open(threshold_path, "r", encoding="utf-8") as f:
            threshold_dict = json.load(f)

        thresholds = torch.tensor(
            [threshold_dict.get(label, 0.5) for label in labels],
            dtype=torch.float32,
            device=device,
        )
    else:
        thresholds = torch.full((len(labels),), 0.5, device=device)
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
async def predict(file: UploadFile = File(...), top_k: int = 10, max_explanations: int = 5):
    if model is None or val_tf is None or labels is None or thresholds is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    x = val_tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs_tensor = torch.sigmoid(logits)[0]
        preds_tensor = probs_tensor >= thresholds

    probs = probs_tensor.detach().cpu().numpy()
    preds = preds_tensor.detach().cpu().numpy()

    positive_indices = [i for i, pred in enumerate(preds) if pred]

    positive_indices = sorted(
        positive_indices,
        key=lambda i: probs[i],
        reverse=True,
    )

    explained_indices = set(positive_indices[:max_explanations])

    results = []

    for i in positive_indices[:top_k]:
        item = {
            "label": labels[i],
            "probability": float(probs[i]),
            "threshold": float(thresholds[i].detach().cpu()),
            "positive": bool(preds[i]),
        }

        if i in explained_indices:
            item["gradcam"] = generate_gradcam_base64(
                image=image,
                input_tensor=x,
                class_idx=i,
                cfg=model.cfg,
            )

        results.append(item)

    return {
        "filename": file.filename,
        "top_k": top_k,
        "max_explanations": max_explanations,
        "predictions": results,
    }


def get_target_layer(model):
    backbone = model.model

    if hasattr(backbone, "features"):  # DenseNet / ConvNeXt
        return backbone.features[-1]

    if hasattr(backbone, "layer4"):  # ResNet
        return backbone.layer4[-1]

    raise RuntimeError("Grad-CAM target layer not supported for this backbone yet.")


def image_to_base64(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", img_bgr)
    if not ok:
        raise RuntimeError("Could not encode Grad-CAM image")
    return base64.b64encode(buffer).decode("utf-8")


def prepare_display_image(image, cfg):
    resize = cfg.augmentation.resize
    crop = cfg.augmentation.crop

    image = image.convert("RGB")
    image = image.resize((resize, resize))

    left = (resize - crop) // 2
    top = (resize - crop) // 2
    image = image.crop((left, top, left + crop, top + crop))

    return np.array(image).astype(np.float32) / 255.0


def generate_gradcam_base64(image, input_tensor, class_idx, cfg):
    target_layer = get_target_layer(model)

    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
    )

    targets = [ClassifierOutputTarget(class_idx)]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets,
    )[0]

    rgb_img = prepare_display_image(image, cfg)

    overlay = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True,
    )

    return image_to_base64(overlay)