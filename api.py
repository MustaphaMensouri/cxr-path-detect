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
from pytorch_grad_cam import GradCAMPlusPlus
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
    positive_indices = sorted(positive_indices, key=lambda i: probs[i], reverse=True)

    top_indices = positive_indices[:top_k]
    explained_indices = set(top_indices[:max_explanations])

    results = []

    if explained_indices:
        target_layer = get_target_layer(model)

        with GradCAMPlusPlus(
            model=model,
            target_layers=[target_layer],
        ) as cam:
            for i in top_indices:
                item = {
                    "label": labels[i],
                    "probability": float(probs[i]),
                    "threshold": float(thresholds[i].detach().cpu()),
                    "positive": bool(preds[i]),
                }

                if i in explained_indices:
                    heatmap_png, attention_boxes = generate_explanation(
                    cam=cam,
                    input_tensor=x,
                    class_idx=i,
                )

                item["heatmap_png"] = heatmap_png
                item["attention_boxes"] = attention_boxes
                item["explanation_method"] = "gradcam++"
                item["heatmap_mime_type"] = "image/png"

                results.append(item)
    else:
        for i in top_indices:
            results.append({
                "label": labels[i],
                "probability": float(probs[i]),
                "threshold": float(thresholds[i].detach().cpu()),
                "positive": bool(preds[i]),
            })

    return {
        "filename": file.filename,
        "top_k": top_k,
        "max_explanations": max_explanations,
        "predictions": results,
        "preprocessing": {
            "resize": int(model.cfg.augmentation.resize),
            "crop": int(model.cfg.augmentation.crop),
            "type": "resize_then_center_crop",
        },
        "heatmap_size": {
            "width": int(x.shape[-1]),
            "height": int(x.shape[-2]),
        },
    }


def get_target_layer(model):
    backbone = model.model

    if hasattr(backbone, "features"):  # DenseNet / ConvNeXt
        return backbone.features[-1]

    if hasattr(backbone, "layer4"):  # ResNet
        return backbone.layer4[-1]

    raise RuntimeError("Grad-CAM target layer not supported for this backbone yet.")


def heatmap_to_transparent_png_base64(
    grayscale_cam,
    alpha_max=180,
    min_alpha_activation=0.08,
):
    grayscale_cam = np.clip(grayscale_cam, 0.0, 1.0)

    cam_uint8 = np.uint8(255 * grayscale_cam)

    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_bgra = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2BGRA)

    alpha = (alpha_max * grayscale_cam).astype(np.uint8)
    alpha[grayscale_cam < min_alpha_activation] = 0

    heatmap_bgra[:, :, 3] = alpha

    ok, buffer = cv2.imencode(".png", heatmap_bgra)
    if not ok:
        raise RuntimeError("Could not encode transparent Grad-CAM++ heatmap")

    return base64.b64encode(buffer).decode("utf-8")

def generate_explanation(cam, input_tensor, class_idx):
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets,
    )[0]
    heatmap_png = heatmap_to_transparent_png_base64(
        grayscale_cam,
        alpha_max=180,
        min_alpha_activation=0.08,
    )
    attention_boxes = cam_to_attention_boxes(
        grayscale_cam,
        max_boxes=4,
        threshold=0.45,
        min_area=80,
    )

    return heatmap_png, attention_boxes

def cam_to_attention_boxes(
    grayscale_cam,
    max_boxes=4,
    threshold=0.45,
    min_area=80,
):
    cam = np.clip(grayscale_cam, 0.0, 1.0)

    mask = (cam >= threshold).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area < min_area:
            continue

        region_score = float(cam[y:y + h, x:x + w].mean())

        boxes.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "score": region_score,
        })

    boxes = sorted(boxes, key=lambda b: b["score"], reverse=True)

    return boxes[:max_boxes]