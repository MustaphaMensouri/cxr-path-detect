# Chest X-ray Multi-label Pathology Classification

This repository project developed during my internship at **Medios Santé**. The project aiming to assist radiologists by reducing diagnostic time and minimizing human error.

The project is a deep learning pipeline for **multi-label chest X-ray pathology classification** using the PadChest dataset.
It includes data preprocessing, patient-level train/validation/test splitting, model training with PyTorch Lightning, experiment tracking with Weights & Biases, threshold tuning, Grad-CAM visualization, and a FastAPI inference API.

---

## Project Overview

The goal of this project is to classify lung-related findings and pathologies from chest X-ray images.
The pipeline supports:

* Building custom PadChest data marts
* Filtering images by projection type, such as PA, AP, L, or all projections
* Selecting lung-related labels
* Patient-level splitting to reduce data leakage
* Training multi-label classification models
* Handling class imbalance using different loss functions
* Tuning per-label thresholds
* Logging experiments with Weights & Biases
* Exporting trained models as W&B artifacts
* Serving predictions through a FastAPI API
* Returning Grad-CAM heatmaps for model explanations

---

## Project Structure

```text
.
├── api.py                  # FastAPI inference API
├── app.py                  # Optional demo app
├── Dockerfile              # Docker image for API deployment
├── preprocess.py           # PadChest preprocessing and data mart creation
├── train.py                # Training entry point using Hydra and Lightning
├── requirements-api.txt    # API dependencies
├── configs/                # Hydra configuration files
│   ├── config.yaml
│   ├── augmentation/
│   ├── data/
│   ├── loss/
│   └── model/
├── src/
│   ├── datamodule.py       # Dataset and LightningDataModule
│   ├── factories.py        # Model, loss, and transform builders
│   └── lightning_module.py # Lightning model, metrics, and threshold tuning
└── docs/                   # Detailed documentation
```

---

## Main Components

### 1. Preprocessing

`preprocess.py` prepares the PadChest dataset for training.
It can:

* Filter images by projection
* Keep selected labels only
* Remove rare labels
* Create patient-safe train/validation/test splits
* Resize or copy images
* Generate label distribution reports
* Save `train.csv`, `val.csv`, `test.csv`, and `labels_used.txt`

> Custom Dataset Support: You can use any custom dataset with this pipeline. To do so, your data directory must follow this structure:

```
data/
├── images/               # Contains all the CXR images
├── train.csv
├── val.csv
├── test.csv
└── labels_used.txt       # List of target labels (one per line)
```
Each .csv file must contain an image_path column and a binary column (0 or 1) for every label listed in labels_used.txt

More details: [Preprocessing documentation](docs/preprocessing.md)

---

### 2. Training

`train.py` trains the multi-label classifier using:

* PyTorch Lightning
* Hydra configuration
* TorchMetrics
* Weights & Biases logging
* Model checkpointing
* Early stopping
* Per-label threshold tuning

If W&B is enabled, the best model checkpoint and tuned thresholds are saved as a W&B artifact; otherwise, they are saved to the checkpoint directory.

More details: [Training documentation](docs/training.md)

---

### 3. Inference API

`api.py` provides a FastAPI service for model inference.

Available endpoints:

* `GET /` — check if the API is running
* `GET /health` — check model loading status
* `POST /predict` — upload a chest X-ray image and get predictions

The API loads a trained model from a W&B artifact and returns:

* Predicted labels
* Probabilities
* Tuned thresholds
* Positive predictions
* Optional Grad-CAM heatmaps

More details: [API documentation](docs/api.md)

---

### 4. Docker Deployment

The project includes a `Dockerfile` for containerizing the API.

The Docker image can be used to deploy the model on a server or cloud platform.

More details: [Deployment documentation](docs/deployment.md)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/MustaphaMensouri/cxr-path-detect.git
cd cxr-path-detect
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies for the API:

```bash
pip install -r requirements-api.txt
```

For training, install the full training requirements:

```bash
pip install -r requirements-training.txt
```

---

## Dataset Preparation

The preprocess.py designed for the PadChest dataset.

A typical preprocessing command looks like this:

```bash
python preprocess.py \
  --data-root /path/to/padchest \
  --output-dir ./data_marts/padchest_lung_PA \
  --datamart-name lung_PA \
  --projections PA \
  --labels-file lung_labels.txt \
  --image-size 224
```

This creates a data mart containing:

```text
data_marts/padchest_lung_PA/
├── images/
├── train.csv
├── val.csv
├── test.csv
├── labels_used.txt
├── label_map.csv
├── split_summary.csv
└── label_distribution_by_split.csv
```

---

## Training

A basic training command:

```bash
python train.py \
  data.data_dir="/path/to/data_mart" \
  data.labels_path="/path/to/data_mart/labels_used.txt"
```

Example with custom loss and backbone:

```bash
python train.py \
  data.data_dir="/path/to/data_mart" \
  data.labels_path="/path/to/data_mart/labels_used.txt" \
  loss=weighted_bce \
  model.backbone=densenet121
```

Example with sampling enabled:

```bash
python train.py \
  data.data_dir="/path/to/data_mart" \
  data.labels_path="/path/to/data_mart/labels_used.txt" \
  data.sample.enabled=true \
  data.sample.size=10000
```

Training configuration is controlled using Hydra files inside `configs/`.

---

## Configuration

The main configuration file is:

```text
configs/config.yaml
```

It controls:

* Training epochs
* GPU devices
* Precision
* Distributed training strategy
* W&B project settings
* Default data, model, loss, and augmentation configs

Loss configs are stored in:

```text
configs/loss/
```

Available losses include:

* Weighted BCE
* Focal Loss
* Asymmetric Loss
* Combined Loss

Augmentation configs are stored in:

```text
configs/augmentation/
```

---

## Experiment Tracking

This project uses Weights & Biases for experiment tracking.

During training, the pipeline logs:

* Training and validation losses
* Macro and micro AUROC
* Macro and micro F1-score
* Average precision
* Tuned threshold metrics
* Best model checkpoint
* Threshold values

The final W&B artifact contains:

```text
model.ckpt
thresholds.json
labels_used.txt
```

---

## Running the API Locally

Set the W&B artifact to load:

```bash
export WANDB_ARTIFACT="your-entity/your-project/lung-pathology-classifier:production"
```

Run the API:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Check the API:

```bash
curl http://localhost:8000/health
```

Send an image for prediction:

```bash
curl -X POST "http://localhost:8000/predict?top_k=10&max_explanations=5" \
  -F "file=@/path/to/image.jpg"
```

---

## Running with Docker

Build the Docker image:

```bash
docker build -t cxr-api:latest .
```

Run the container:

```bash
docker run -p 8000:8000 \
  -e WANDB_API_KEY="your-wandb-api-key" \
  -e WANDB_ARTIFACT="your-entity/your-project/lung-pathology-classifier:production" \
  cxr-api:latest
```

Then open:

```text
http://localhost:8000/health
```

---

## Evaluation Metrics

The project uses multi-label classification metrics, including:

* Macro AUROC
* Micro AUROC
* Macro F1-score
* Micro F1-score
* Macro precision
* Macro recall
* Average precision
* Per-label metrics on the test set
* Tuned-threshold F1 metrics

Because this is a highly imbalanced multi-label medical imaging task, AUC and F1-score should be interpreted together.

More details: [Evaluation documentation](docs/evaluation.md)

---

## Limitations

Important limitations:

* PadChest labels are derived from radiology reports, so they are weak image-level labels.
* Some labels were manually annotated, while others were automatically extracted.
* The dataset has strong class imbalance.
* Some pathologies are rare and difficult to learn.
* A study can contain multiple images sharing the same report labels.
* The model has not been clinically validated yet.

---

## Documentation

Detailed documentation is available in the `docs/` folder:

* [Dataset](docs/dataset.md)
* [Preprocessing](docs/preprocessing.md)
* [Training](docs/training.md)
* [Evaluation](docs/evaluation.md)
* [API](docs/api.md)
* [Deployment](docs/deployment.md)
* [Experiments](docs/experiments.md)

---

## Future Work

Possible improvements:

* Add a simple web interface for image upload and Grad-CAM visualization
* Compare more backbones such as DenseNet, ResNet, ConvNeXt, Swin, and ViT
* Improve calibration and threshold selection
* Add external validation on another chest X-ray dataset
* Add model cards for each trained model
* Add CI/CD for automatic Docker image builds
* Improve explainability and reporting
* Add better documentation for experiments and results

---

## License

MIT License

---

## Author

Mustapha Mensouri

Master 2 Artificial Intelligence
Chest X-ray pathology classification project