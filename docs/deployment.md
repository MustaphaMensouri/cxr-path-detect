# Deployment Documentation

This document explains how to deploy the Chest X-ray classifier API using Docker and Weights & Biases artifacts.

For API endpoint details, see:

```text
docs/api.md
```

For training and artifact creation details, see:

```text
docs/training.md
```

---

## 1. Purpose

The goal of deployment is to run the trained chest X-ray classification model as an API service.

The deployment uses:

* FastAPI for inference
* Docker for containerization
* Weights & Biases artifacts for model storage
* Uvicorn as the API server

The Docker image contains the API code, but the model weights are loaded from W&B when the container starts.

---

## 2. Deployment Overview

The deployment workflow is:

```text
Train model
   ↓
Save model artifact to W&B
   ↓
Build Docker image
   ↓
Run Docker container
   ↓
API downloads model from W&B
   ↓
Send image requests to /predict
```

The model is not copied directly into the Docker image.

Instead, the API loads the model artifact using the environment variable:

```text
WANDB_ARTIFACT
```

---

## 3. Deployment Requirements

Before deployment, make sure you have:

* Docker installed
* A trained model saved as a W&B artifact
* A W&B API key
* Internet access from the deployment machine
* The project source code
* A valid `requirements-api.txt` file

The container needs internet access because it downloads the model artifact from W&B at startup.

---

## 4. Model Artifact Requirement

The deployed model must be available as a W&B artifact.

The artifact should contain:

```text
model.ckpt
thresholds.json
labels_used.txt
```

### `model.ckpt`

This is the trained PyTorch Lightning checkpoint.

It contains:

* model weights
* model configuration
* number of classes
* class names

### `thresholds.json`

This file contains the tuned threshold for each label.

If this file is missing, the API uses `0.5` as the threshold for all labels.

### `labels_used.txt`

This file contains the label names used during training.

It is included for reproducibility.

---

## 5. Environment Variables

The deployment uses environment variables to configure W&B and model loading.

### `WANDB_API_KEY`

This allows the container to access W&B.

Example:

```bash
export WANDB_API_KEY="wandb_api_key"
```

### `WANDB_ARTIFACT`

This tells the API which model artifact to load.

Example:

```bash
export WANDB_ARTIFACT="username_wandb/lung-pathology-multilabel/lung-pathology-classifier:production"
```

The artifact can point to a specific version:

```text
username_wandb/lung-pathology-multilabel/lung-pathology-classifier:v2
```

Or to an alias:

```text
username_wandb/lung-pathology-multilabel/lung-pathology-classifier:production
```

Using an alias such as `production` is useful because the deployed API can load the current production model without changing the code.

---

## 6. Docker Image

The Docker image is defined by:

```text
Dockerfile
```

The Docker image:

1. Uses Python 3.11 slim.
2. Installs system libraries required by OpenCV.
3. Installs API dependencies from `requirements-api.txt`.
4. Copies `api.py`.
5. Copies the `src/` folder.
6. Exposes port `8000`.
7. Starts the API using Uvicorn.

The image contains the API code only.
The model is downloaded from W&B when the API starts.

---

## 7. Build the Docker Image

From the project root, run:

```bash
docker build -t cxr-api:latest .
```

This creates a local Docker image named:

```text
cxr-api:latest
```

---

## 8. Run the Container Locally

Run the API container:

```bash
docker run -p 8000:8000 \
  -e WANDB_API_KEY="wandb_api_key" \
  -e WANDB_ARTIFACT="username_wandb/lung-pathology-multilabel/lung-pathology-classifier:production" \
  cxr-api:latest
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

## 9. Test the Running Container

First, check if the API is running:

```bash
curl http://localhost:8000/
```

Then check if the model is loaded:

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "device": "cpu",
  "num_labels": 55,
  "artifact": "username_wandb/lung-pathology-multilabel/lung-pathology-classifier:production"
}
```

Then test prediction:

```bash
curl -X POST "http://localhost:8000/predict?top_k=10&max_explanations=5" \
  -F "file=@/path/to/chest_xray.jpg"
```

The API returns positive labels whose probability is greater than or equal to their threshold.

---

## 10. Run the Container in the Background

To run the container in detached mode:

```bash
docker run -d \
  --name cxr-api \
  -p 8000:8000 \
  -e WANDB_API_KEY="wandb_api_key" \
  -e WANDB_ARTIFACT="username_wandb/lung-pathology-multilabel/lung-pathology-classifier:production" \
  cxr-api:latest
```

Check logs:

```bash
docker logs -f cxr-api
```

Stop the container:

```bash
docker stop cxr-api
```

Remove the container:

```bash
docker rm cxr-api
```

---

## 11. Push the Image to Docker Hub

Tag the image:

```bash
docker tag cxr-api:latest dockerhub_username/cxr-api:latest
```

Push the image:

```bash
docker push dockerhub_username/cxr-api:latest
```

After pushing, the image can be pulled from another machine or server.

---

## 12. Run on a Server

On the server, pull the image:

```bash
docker pull dockerhub_username/cxr-api:latest
```

Run the container:

```bash
docker run -d \
  --name cxr-api \
  -p 8000:8000 \
  -e WANDB_API_KEY="wandb_api_key" \
  -e WANDB_ARTIFACT="username_wandb/lung-pathology-multilabel/lung-pathology-classifier:production" \
  dockerhub_username/cxr-api:latest
```

Check the logs:

```bash
docker logs -f cxr-api
```

Check the health endpoint:

```bash
curl http://server-ip:8000/health
```

If the server has a firewall, make sure port `8000` is open.

---

## 13. Updating the Deployed Model

There are two common update cases.

---

### Case 1: Update only the model

If the API code did not change and only the model changed:

1. Train a new model.
2. Save it as a W&B artifact.
3. Move the `production` alias to the new artifact version.
4. Restart the container.

Example:

```bash
docker restart cxr-api
```

When the container restarts, the API downloads the artifact pointed to by:

```text
WANDB_ARTIFACT
```

If `WANDB_ARTIFACT` uses the `production` alias, the API will load the new production model.

---

### Case 2: Update the API code

If `api.py`, `src/`, or dependencies changed:

1. Rebuild the Docker image.
2. Push the new image to Docker Hub.
3. Pull the new image on the server.
4. Stop and remove the old container.
5. Run the new container.

Example:

```bash
docker build -t dockerhub_username/cxr-api:latest .
docker push dockerhub_username/cxr-api:latest
```

On the server:

```bash
docker pull dockerhub_username/cxr-api:latest

docker stop cxr-api
docker rm cxr-api

docker run -d \
  --name cxr-api \
  -p 8000:8000 \
  -e WANDB_API_KEY="wandb_api_key" \
  -e WANDB_ARTIFACT="username_wandb/lung-pathology-multilabel/lung-pathology-classifier:production" \
  dockerhub_username/cxr-api:latest
```

---

## 14. Suggested Production Workflow

A recommended workflow is:

```text
1. Train several experiments.
2. Compare validation and test results.
3. Select the best model.
4. Save the model as a W&B artifact.
5. Promote the selected artifact to the production alias.
6. Build and run the Docker API container.
7. Test /health.
8. Test /predict with sample images.
9. Monitor logs.
```

This keeps the API code and model version separate.

---

## 15. Common Deployment Issues

### W&B authentication error

Problem:

```text
The container cannot access W&B.
```

Possible fixes:

* Check that `WANDB_API_KEY` is set.
* Check that the API key is valid.
* Check that the W&B artifact exists.
* Check that the W&B project is accessible by the account.

---

### Artifact not found

Problem:

```text
The API cannot find the W&B artifact.
```

Possible fixes:

* Check the value of `WANDB_ARTIFACT`.
* Make sure the entity, project, artifact name, and alias/version are correct.
* Make sure the artifact was logged successfully during training.

---

### Model not loaded

If `/health` returns:

```json
{
  "status": "loading"
}
```

The model may still be downloading or loading.

Check logs:

```bash
docker logs -f cxr-api
```

If loading fails, the logs should show the error.

---

### `/predict` returns 503

This means the model is not loaded yet.

Wait for startup to finish, then check:

```bash
curl http://localhost:8000/health
```

---

### `/predict` returns 422

This usually means the image was not sent with the correct form-data key.

The key must be named:

```text
file
```

Correct curl example:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.jpg"
```

---

### Port already in use

If port `8000` is already used, either stop the other process or map the container to another host port.

Example:

```bash
docker run -p 8080:8000 cxr-api:latest
```

Then use:

```text
http://localhost:8080
```

---

### CPU inference is slow

If no GPU is available, the API runs on CPU.

This is expected, but inference and Grad-CAM may be slower.

---

### Grad-CAM is slow

Grad-CAM requires an additional backward pass.

To reduce response time, use:

```text
max_explanations=0
```

or a small value such as:

```text
max_explanations=1
```

---
