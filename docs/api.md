# API Documentation

This document explains how the FastAPI inference API works, how to run it locally, how to send prediction requests, and how to interpret the response.

The API is implemented in:

```text
api.py
```

It serves a trained chest X-ray multi-label classification model.

---

## 1. Purpose

The API allows a user or application to upload a chest X-ray image and receive model predictions.

For each uploaded image, the API returns:

* predicted labels
* probabilities
* tuned thresholds
* positive predictions
* optional Grad-CAM explanations

This API is intended for research and demonstration purposes only. It is not a medical diagnosis system.

---

## 2. API Overview

The API is built with FastAPI.

Available endpoints:

| Method | Endpoint   | Purpose                             |
| ------ | ---------- | ----------------------------------- |
| `GET`  | `/`        | Basic API status message            |
| `GET`  | `/health`  | Check model loading status          |
| `POST` | `/predict` | Upload an image and get predictions |

---

## 3. Model Loading

The model is loaded automatically when the API starts.

At startup, the API:

1. Reads the W&B artifact path from `WANDB_ARTIFACT`.
2. Downloads the model artifact from Weights & Biases.
3. Loads `model.ckpt`.
4. Reads the saved class names from the checkpoint.
5. Loads `thresholds.json` if it exists.
6. Rebuilds the model architecture.
7. Loads the model weights.
8. Builds the validation transform.
9. Moves the model to GPU if available, otherwise CPU.

The API expects the W&B artifact to contain:

```text
model.ckpt
thresholds.json
labels_used.txt
```

The most important file is:

```text
model.ckpt
```

The API reads the model configuration and class names from this checkpoint.

---

## 4. Environment Variables

The API uses the following environment variables.

### `WANDB_ARTIFACT`

This tells the API which model artifact to load.

Example:

```bash
export WANDB_ARTIFACT="username_wandb/lung-pathology-multilabel/lung-pathology-classifier:production"
```

If this variable is not set, the API uses the default value defined in `api.py`.

### `WANDB_API_KEY`

This is required when the W&B artifact is private or when running the API inside Docker or on a server.

Example:

```bash
export WANDB_API_KEY="wandb_api_key"
```

---

## 5. Running the API Locally

Install the API dependencies:

```bash
pip install -r requirements-api.txt
```

Start the API:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI documentation will be available at:

```text
http://localhost:8000/docs
```

---

## 6. Health Check

Use the `/health` endpoint to check if the model is loaded.

Request:

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "device": "cuda",
  "num_labels": 55,
  "artifact": "mustaphamensouri/lung-pathology-multilabel/lung-pathology-classifier:production"
}
```

Response fields:

| Field        | Meaning                                          |
| ------------ | ------------------------------------------------ |
| `status`     | `ok` if the model is loaded, otherwise `loading` |
| `device`     | device used by the API, usually `cuda` or `cpu`  |
| `num_labels` | number of labels predicted by the model          |
| `artifact`   | W&B artifact loaded by the API                   |

---

## 7. Prediction Endpoint

The prediction endpoint is:

```text
POST /predict
```

It expects an image file uploaded as form-data.

### Query Parameters

| Parameter          | Default | Meaning                                           |
| ------------------ | ------- | ------------------------------------------------- |
| `top_k`            | `10`    | maximum number of positive predictions to return  |
| `max_explanations` | `5`     | maximum number of Grad-CAM explanations to return |

Important note:

The API returns only labels where:

```text
probability >= threshold
```

Then it sorts those positive labels by probability and returns up to `top_k`.

---

## 8. Example Prediction Request

Using `curl`:

```bash
curl -X POST "http://localhost:8000/predict?top_k=10&max_explanations=5" \
  -F "file=@/path/to/chest_xray.jpg"
```

The form-data field name must be:

```text
file
```

If the field is missing or named incorrectly, FastAPI will return a validation error.

---

## 9. Example Prediction Response

Example response:

```json
{
  "filename": "chest_xray.jpg",
  "top_k": 10,
  "max_explanations": 5,
  "predictions": [
    {
      "label": "chronic_changes",
      "probability": 0.8137,
      "threshold": 0.4200,
      "positive": true,
      "gradcam": "base64_encoded_image_string"
    },
    {
      "label": "fibrotic_band",
      "probability": 0.6780,
      "threshold": 0.3500,
      "positive": true
    }
  ]
}
```

Response fields:

| Field              | Meaning                                           |
| ------------------ | ------------------------------------------------- |
| `filename`         | uploaded image filename                           |
| `top_k`            | requested maximum number of predictions           |
| `max_explanations` | requested maximum number of Grad-CAM explanations |
| `predictions`      | list of positive predictions                      |

Each prediction contains:

| Field         | Meaning                            |
| ------------- | ---------------------------------- |
| `label`       | predicted class name               |
| `probability` | model probability after sigmoid    |
| `threshold`   | threshold used for this label      |
| `positive`    | whether the prediction is positive |
| `gradcam`     | optional base64 Grad-CAM image     |

The `gradcam` field appears only for the first `max_explanations` predictions.

---

## 10. Thresholds

The API uses one threshold per label.

Thresholds are loaded from:

```text
thresholds.json
```

If `thresholds.json` exists in the W&B artifact, the API loads it.

If a label is missing from `thresholds.json`, the API uses:

```text
0.5
```

If `thresholds.json` does not exist, the API uses `0.5` for all labels.

Thresholds are important because this is an imbalanced multi-label classification problem. A default threshold of `0.5` is not always the best choice for every label.

---

## 11. Grad-CAM Explanations

The API can return Grad-CAM heatmaps for some predictions.

Grad-CAM is used to highlight image regions that influenced the prediction.

The API returns Grad-CAM as a base64-encoded JPEG string:

```json
{
  "gradcam": "base64_encoded_image_string"
}
```

To view the Grad-CAM image, the base64 string must be decoded into an image.

Current Grad-CAM support depends on the model backbone.

Supported target layers:

| Backbone type       | Target layer            |
| ------------------- | ----------------------- |
| DenseNet / ConvNeXt | `backbone.features[-1]` |
| ResNet              | `backbone.layer4[-1]`   |

If the backbone does not have one of these structures, Grad-CAM may not be supported.

---

## 12. Testing with Postman

To test the API with Postman:

1. Set method to `POST`.
2. Use this URL:

```text
http://localhost:8000/predict?top_k=10&max_explanations=5
```

3. Go to the `Body` tab.
4. Select `form-data`.
5. Add a key named:

```text
file
```

6. Change the key type from `Text` to `File`.
7. Select a `.jpg` or `.png` chest X-ray image.
8. Click `Send`.

Correct Postman setup:

| Key    | Type | Value          |
| ------ | ---- | -------------- |
| `file` | File | selected image |

If you get a `422` error saying that `file` is missing, it usually means the request was not sent as form-data with the key named `file`.

---

## 13. Error Responses

### Model not loaded

Status code:

```text
503
```

Example:

```json
{
  "detail": "Model is not loaded yet"
}
```

This happens if the `/predict` endpoint is called before the model finishes loading.

---

### Invalid image file

Status code:

```text
400
```

Example:

```json
{
  "detail": "Invalid image file"
}
```

This happens if the uploaded file cannot be opened as an image.

---

### Missing file field

Status code:

```text
422
```

Example:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "file"],
      "msg": "Field required"
    }
  ]
}
```

This happens when the request does not include a form-data field named `file`.
