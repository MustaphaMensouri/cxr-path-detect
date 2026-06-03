# Training Documentation

This document explains how the training pipeline works, how to configure experiments, how data is loaded, how metrics are computed, and how trained models are saved.

The training pipeline is designed for multi-label chest X-ray classification using PyTorch Lightning, Hydra configuration files, and optional Weights & Biases experiment tracking.

---

## 1. Purpose

The goal of the training pipeline is to train a multi-label chest X-ray classifier that predicts multiple lung-related findings or pathologies from one X-ray image.

The training pipeline supports:

* Multi-label classification
* Configurable CNN/vision backbones from `torchvision`
* Multiple loss functions for class imbalance
* Data augmentation
* Optional patient-level training subset sampling
* Validation metrics during training
* Per-label threshold tuning
* Optional test evaluation
* W&B experiment tracking
* W&B model artifact export

---

## 2. Training Entry Point

The main training entry point is:

```bash
python train.py
```

The `train.py` script is responsible for:

1. Loading the Hydra configuration.
2. Setting the random seed.
3. Creating the `XrayDataModule`.
4. Loading the class names from `labels_used.txt`.
5. Creating the `XrayClassifier`.
6. Setting up W&B or local CSV logging.
7. Defining callbacks such as checkpointing, early stopping, learning-rate monitoring, and progress bar.
8. Running `trainer.fit(...)`.
9. Saving the best model and tuned thresholds as a W&B artifact.
10. Optionally running test evaluation.

The training script uses Hydra, so most parameters can be changed from the command line without editing the source code.

---

## 3. Input Data Requirements

The training pipeline expects a prepared data mart directory.

A typical data mart should contain:

```text
data_mart/
├── images/
├── train.csv
├── val.csv
├── test.csv
└── labels_used.txt
```

### Required CSV files

The following files are expected:

```text
train.csv
val.csv
test.csv
```

Each CSV should contain:

* `image_path`: relative path to the image file
* one column per label, with binary values `0` or `1`

Example:

```text
image_path,normal,pleural_effusion,pneumothorax,atelectasis
images/example_001.jpg,0,1,0,1
```

### Required labels file

The file `labels_used.txt` should contain one label column name per line.

Example:

```text
normal
pleural_effusion
pneumothorax
atelectasis
```

The labels in `labels_used.txt` must match column names in the CSV files.

---

## 4. Configuration System

The project uses Hydra for configuration.

The main configuration file is:

```text
configs/config.yaml
```

It loads default configuration groups:

```yaml
defaults:
  - data: default
  - model: default
  - loss: weighted_bce
  - augmentation: light
  - _self_
```

This means the final training configuration is composed from:

```text
configs/config.yaml
configs/data/default.yaml
configs/model/default.yaml
configs/loss/*.yaml
configs/augmentation/*.yaml
```

The main config controls:

* number of epochs
* accelerator
* number of devices
* distributed training strategy
* precision
* W&B settings
* selected data config
* selected model config
* selected loss config
* selected augmentation config

Example training settings:

```yaml
train:
  max_epochs: 30
  accelerator: auto
  devices: 2
  strategy: ddp
  precision: 16-mixed
  log_every_n_steps: 50
  enable_progress_bar: true
  enable_model_summary: true
  run_test: false
```

---

## 5. Training With and Without W&B

The pipeline can run with or without Weights & Biases.

### With W&B

By default, W&B is enabled:

```yaml
wandb:
  enabled: true
```

When W&B is enabled, the training script:

* creates a W&B run
* logs metrics
* logs the resolved configuration
* saves the best model artifact
* saves `thresholds.json`
* saves `labels_used.txt`

Example:

```bash
python train.py \
  wandb.enabled=true
```

### Without W&B

To disable W&B:

```bash
python train.py \
  wandb.enabled=false
```

When W&B is disabled, the project uses a local CSV logger instead.

Local logs are saved under:

```text
logs/local_run/
```

This is useful for debugging or training in environments where W&B login is not available.

---

## 6. Model Architecture and Loss Functions

The model is implemented in:

```text
src/lightning_module.py
```

The model class is:

```python
XrayClassifier
```

The classifier uses a backbone created by:

```text
src/factories.py
```

The backbone is selected from `torchvision.models` using the value of:

```yaml
model.backbone
```

The final classification layer is replaced so that the model outputs one logit per label.

For example, if the dataset has 55 labels, the final layer outputs 55 logits.

---

### Supported backbone types

The code supports common `torchvision` models with one of these head types:

* `classifier`
* `fc`
* `heads`

This allows models such as:

* DenseNet
* ResNet
* ConvNeXt
* Vision Transformer models available in `torchvision`

The exact model name must be available in `torchvision.models`.

Example:

```bash
python train.py \
  model.backbone=densenet121
```

---

### Loss functions

Loss functions are built in:

```text
src/factories.py
```

Available loss configs are stored in:

```text
configs/loss/
```

Supported losses:

```text
weighted_bce
focal
asl
combined
```

### Weighted BCE

Config:

```yaml
name: weighted_bce
max_weight: 10.0
```

Weighted BCE is used to reduce the impact of class imbalance by giving higher weight to rare positive labels.

Example:

```bash
python train.py loss=weighted_bce
```

### Focal Loss

Config:

```yaml
name: focal
alpha: 0.25
gamma: 2.0
reduction: mean
```

Focal Loss focuses more on difficult examples and reduces the contribution of easy examples.

Example:

```bash
python train.py loss=focal
```

### Asymmetric Loss

Config:

```yaml
name: asl
gamma_neg: 4.0
gamma_pos: 1.0
clip: 0.05
```

Asymmetric Loss is designed for imbalanced multi-label classification. It applies different focusing behavior to positive and negative labels.

Example:

```bash
python train.py loss=acl
```

### Combined Loss

Config:

```yaml
name: combined
losses:
  - name: focal
  - name: weighted_bce
weights: [0.5, 0.5]
```

Combined loss allows multiple losses to be used together.

Example:

```bash
python train.py loss=combined
```

---

## 7. Data Loading and Sampling

Data loading is implemented in:

```text
src/datamodule.py
```

The main classes are:

```python
XrayDataset
XrayDataModule
```

### XrayDataset

`XrayDataset`:

1. Reads a split CSV file.
2. Loads image paths from the `image_path` column.
3. Opens each image with PIL.
4. Converts images to RGB.
5. Applies the selected transform.
6. Reads the target labels from the label columns.
7. Returns image tensor and multi-label target tensor.

Each sample returns:

```python
image, label
```

Where:

* `image` is a transformed image tensor
* `label` is a multi-hot tensor of shape `[num_classes]`

---

### XrayDataModule

`XrayDataModule` creates three dataloaders:

```python
train_dataloader()
val_dataloader()
test_dataloader()
```

The training dataloader uses training transforms and shuffling.

The validation and test dataloaders use validation transforms without shuffling.

---

### Optional training subset sampling

The training dataset supports optional sampling through:

```yaml
data.sample.enabled=true
```

This is useful when the full training set is large and experiments need to run faster.

The sampling logic:

1. Starts from `train.csv`.
2. Finds rare labels using `rare_threshold`.
3. Keeps all patients that have at least one rare label.
4. Fills the remaining sample budget using patient-level stratified sampling from common-label patients.
5. Prints detailed sampling information to the terminal.

Example:

```bash
python train.py \
  data.sample.enabled=true \
  data.sample.size=10000 \
  data.sample.rare_threshold=500
```

Important note: because sampling is patient-based, the final number of images may not be exactly equal to the requested sample size.

---

## 8. Data Augmentation

Augmentation configs are stored in:

```text
configs/augmentation/
```

Available configs:

```text
light.yaml
strong.yaml
```

The training transform includes:

* resize
* random crop
* random affine transformation
* random autocontrast
* tensor conversion
* ImageNet normalization

The validation and test transform includes:

* resize
* center crop
* tensor conversion
* ImageNet normalization

Example using light augmentation:

```bash
python train.py augmentation=light
```

Example using strong augmentation:

```bash
python train.py augmentation=strong
```

---

## 9. Training Workflow

The training workflow is:

1. Hydra loads the configuration.
2. The random seed is fixed.
3. The data module loads labels from `labels_used.txt`.
4. The model is created with the selected backbone and loss.
5. A logger is created:

   * W&B logger if `wandb.enabled=true`
   * CSV logger if `wandb.enabled=false`
6. Lightning callbacks are created.
7. `trainer.fit(model, dm)` starts training.
8. At each epoch:

   * training loss is computed
   * validation loss and metrics are computed
   * validation probabilities are collected
   * per-label thresholds are tuned
   * tuned validation metrics are logged
9. The best checkpoint is saved based on `val/f1_macro_tuned`.
10. If W&B is enabled, the best checkpoint, labels file, and thresholds file are saved as an artifact.
11. If `train.run_test=true`, the test set is evaluated using the best checkpoint.

---

## 10. Metrics

The project uses multi-label metrics from TorchMetrics.

Validation and test global metrics include:

```text
auc_macro
auc_micro
f1_macro
f1_micro
precision_macro
recall_macro
ap_macro
ap_micro
```

### Macro metrics

Macro metrics compute the metric separately for each label and then average across labels.

Macro metrics are useful when rare labels matter.

### Micro metrics

Micro metrics combine all label predictions globally before computing the metric.

Micro metrics are more influenced by frequent labels.

### Average Precision

Average Precision is also logged as:

```text
ap_macro
ap_micro
```

This is useful for imbalanced multi-label classification because it summarizes precision-recall behavior across thresholds.

---

## 11. Threshold Tuning

The model outputs raw logits.

During validation and inference, logits are converted to probabilities using sigmoid:

```python
probs = sigmoid(logits)
```

A default threshold of `0.5` is often not optimal for imbalanced multi-label classification.

This project tunes one threshold per label on the validation set.

Threshold candidates are searched from:

```text
0.01 to 0.99
```

For each label, the threshold with the best F1-score is selected.

The tuned thresholds are stored in:

```python
best_thresholds
```

The tuned validation metrics are logged as:

```text
val/f1_macro_tuned
val/f1_micro_tuned
val/precision_macro_tuned
val/recall_macro_tuned
```

During test evaluation, the same stored thresholds are used to compute:

```text
test/f1_macro_tuned
test/f1_micro_tuned
test/precision_macro_tuned
test/recall_macro_tuned
```

The tuned threshold values are also logged per label.

---

## 12. Checkpointing and Early Stopping

The training script uses a model checkpoint callback.

The best checkpoint is selected using:

```text
val/f1_macro_tuned
```

The checkpoint mode is:

```text
max
```

This means the checkpoint with the highest tuned macro F1-score is kept.

The saved checkpoint filename is:

```text
best.ckpt
```

Early stopping also monitors:

```text
val/f1_macro_tuned
```

Default patience:

```text
3
```

This means training stops if the monitored metric does not improve for 3 validation checks.

The patience can be changed with:

```bash
python train.py train.early_stopping_patience=5
```

---

## 13. Optimizer and Scheduler

The optimizer is:

```text
AdamW
```

The learning rate is taken from:

```yaml
model.lr
```

The weight decay is taken from:

```yaml
model.weight_decay
```

The scheduler is:

```text
CosineAnnealingLR
```

The scheduler uses:

```python
T_max = max_epochs
```

This means the learning rate follows a cosine schedule across the full training run.

---

## 14. W&B Logging and Artifacts

When W&B is enabled, the training script logs:

* configuration
* train loss
* validation loss
* validation metrics
* tuned threshold metrics
* learning rate
* checkpoint metadata

After training, the best model is saved as a W&B artifact.

The artifact contains:

```text
model.ckpt
thresholds.json
labels_used.txt
```

### model.ckpt

The checkpoint contains:

* model weights
* saved hyperparameters
* number of classes
* class names
* training configuration

### thresholds.json

This file contains one tuned threshold per label.

Example:

```json
{
  "pleural_effusion": 0.31,
  "pneumothorax": 0.44,
  "atelectasis": 0.22
}
```

### labels_used.txt

This file contains the label names used by the model.

It is saved with the artifact to make inference reproducible.

The artifact is logged with the alias:

```text
candidate
```

Later, a selected model can be promoted manually in W&B by adding an alias such as:

```text
production
```

---

## 15. Test Evaluation

Test evaluation is optional.

By default:

```yaml
train:
  run_test: false
```

To run test evaluation after training:

```bash
python train.py train.run_test=true
```

The test evaluation uses the best checkpoint:

```python
ckpt_path="best"
```

During test evaluation, the project logs:

* global test metrics
* per-label test metrics
* tuned test metrics
* tuned per-label F1
* tuned per-label precision
* tuned per-label recall
* threshold values per label

Test metrics are useful for final model reporting, but they should not be used repeatedly for model selection.

---

## 16. Example Commands

### Basic training

```bash
python train.py \
  data.data_dir="/path/to/data_mart" \
  data.labels_path="/path/to/data_mart/labels_used.txt"
```

### Train with Weighted BCE

```bash
python train.py \
  data.data_dir="/path/to/data_mart" \
  data.labels_path="/path/to/data_mart/labels_used.txt" \
  loss=weighted_bce
```

### Train with Focal Loss

```bash
python train.py \
  data.data_dir="/path/to/data_mart" \
  data.labels_path="/path/to/data_mart/labels_used.txt" \
  loss=focal
```

### Train with Asymmetric Loss

```bash
python train.py \
  data.data_dir="/path/to/data_mart" \
  data.labels_path="/path/to/data_mart/labels_used.txt" \
  loss=acl
```

### Train with a different backbone

```bash
python train.py \
  data.data_dir="/path/to/data_mart" \
  data.labels_path="/path/to/data_mart/labels_used.txt" \
  model.backbone=densenet121
```

### Train with strong augmentation

```bash
python train.py \
  data.data_dir="/path/to/data_mart" \
  data.labels_path="/path/to/data_mart/labels_used.txt" \
  augmentation=strong
```

### Train on a 10k sampled subset

```bash
python train.py \
  data.data_dir="/path/to/data_mart" \
  data.labels_path="/path/to/data_mart/labels_used.txt" \
  data.sample.enabled=true \
  data.sample.size=10000
```

### Train without W&B

```bash
python train.py \
  data.data_dir="/path/to/data_mart" \
  data.labels_path="/path/to/data_mart/labels_used.txt" \
  wandb.enabled=false
```

### Train and run test evaluation

```bash
python train.py \
  data.data_dir="/path/to/data_mart" \
  data.labels_path="/path/to/data_mart/labels_used.txt" \
  train.run_test=true
```

---

## 17. Output Files

### When W&B is enabled

The main output is a W&B artifact containing:

```text
model.ckpt
thresholds.json
labels_used.txt
```

### When W&B is disabled

Local logs are saved under:

```text
logs/local_run/
```

Lightning checkpoint files may also be saved in the current run directory depending on the active logger and working directory.

---

## 18. Common Issues

### 1. Label names do not match CSV columns

If `labels_used.txt` contains labels that are not columns in the CSV, those labels will be ignored by the dataset.

Check that every label in:

```text
labels_used.txt
```

exists as a column in:

```text
train.csv
val.csv
test.csv
```

---

### 2. Image path not found

The dataset loads images using:

```text
data.data_dir / image_path
```

So if `data.data_dir` is:

```text
/path/to/data_mart
```

and `image_path` is:

```text
images/example.jpg
```

then the image must exist at:

```text
/path/to/data_mart/images/example.jpg
```

---

### 3. W&B login error

If W&B is enabled, make sure you are logged in:

```bash
wandb login
```

Or disable W&B:

```bash
python train.py wandb.enabled=false
```

---

### 4. DDP issues on one GPU

The default config may use:

```yaml
devices: 2
strategy: ddp
```

If you are training on one GPU, override it:

```bash
python train.py \
  train.devices=1 \
  train.strategy=auto
```

---

### 5. Out of memory

If you get CUDA out-of-memory errors, try:

```bash
python train.py \
  data.batch_size=16
```

You can also reduce image crop size or use a smaller backbone.

---

### 6. Sampling size is not exact

The sampling logic works at the patient level, not image level.

Because one patient can have multiple images, the final sampled image count can be slightly different from the requested size.

This is expected.

---

### 7. Very low F1 but high AUC

This can happen in imbalanced multi-label classification.

AUC measures ranking quality across thresholds, while F1 depends on the selected threshold.

This is why the project includes per-label threshold tuning.

---

## 19. Reproducibility Notes

The training script sets the random seed to:

```python
42
```

This helps make experiments more reproducible.

However, exact reproducibility is not always guaranteed when using:

* GPU operations
* multi-worker dataloaders
* mixed precision
* distributed training
* random data augmentation

---

## 20. Recommended Experiment Naming

A good experiment name should include:

* dataset version
* projection type
* backbone
* loss function
* sample size
* augmentation type

Example W&B notes:

```text
PA-only PadChest, DenseNet121, weighted BCE, light augmentation, full train set
```

Example tags:

```yaml
wandb:
  tags:
    - padchest
    - PA
    - densenet121
    - weighted_bce
    - light_aug
```

Example 

```bash
!python train.py \
data.data_dir="/path/data mart/" \
data.labels_path="/path/to/labels_used.txt" \
model.backbone=convnext_tiny \
wandb.enabled=true \
"wandb.notes='PA-only PadChest, DenseNet121, weighted BCE, light augmentation, full train set'" \
"wandb.tags=[padchest,densenet121,weighted_bce,light_aug]"
```

---

## 21. Best Model Selection

The current training pipeline selects the best model using:

```text
val/f1_macro_tuned
```

This choice is made because of rare labels matter and macro F1 gives each label equal importance.
