# app.py
import torch
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf

from src.lightning_module import XrayClassifier
from src.factories import build_transforms
from src.datamodule import load_labels

import wandb


CKPT_PATH = "checkpoints/best.ckpt"   # change this
CONFIG_PATH = "/kaggle/working/cxr-path-detect/configs/config.yaml"   # change if needed
LABELS_PATH = "/kaggle/working/data_marts/lung_PA_AP_AP_horizontal/labels_used.txt"


device = "cuda" if torch.cuda.is_available() else "cpu"

api = wandb.Api()
artifact = api.artifact("mustaphamensouri/lung-pathology-multilabel/model-2loyqw63:v0")
artifact_dir = artifact.download()

cfg = OmegaConf.load(CONFIG_PATH)
labels = load_labels(LABELS_PATH)

ckpt_path = f"{artifact_dir}/model.ckpt"  # or best.ckpt

model = XrayClassifier.load_from_checkpoint(
    ckpt_path,
    cfg=cfg,
    num_classes=len(labels),
    class_names=labels,
    max_epochs=cfg.train.max_epochs,
    map_location=device,
)

model.eval()
model.to(device)

_, val_tf = build_transforms(cfg.augmentation)


@torch.no_grad()
def predict(image: Image.Image):
    image = image.convert("RGB")
    x = val_tf(image).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    results = {
        label: float(prob)
        for label, prob in zip(labels, probs)
    }

    # Show highest probabilities first
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    return results


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload chest X-ray"),
    outputs=gr.Label(num_top_classes=10, label="Predicted labels"),
    title="Chest X-ray Multi-label Classifier",
    description=(
        "Upload a chest X-ray image. The model returns the most likely PadChest labels. "
        "This is for research/demo use only, not medical diagnosis."
    ),
)

if __name__ == "__main__":
    demo.launch(share=True)