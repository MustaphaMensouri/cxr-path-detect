# Evaluation Documentation

This document explains how the model is evaluated and why this evaluation strategy is used.

---

## 1. Purpose

The goal of evaluation is to measure how well the model predicts multiple chest X-ray labels.

This is a **multi-label classification** problem, which means one image can have more than one positive label at the same time.

Example:

```text
pleural_effusion = 1
atelectasis = 1
pneumothorax = 0
normal = 0
```

Because of this, the model does not choose only one class.
It outputs one probability for each label.

---

## 2. Validation and Test Sets

The dataset is split into:

```text
train.csv
val.csv
test.csv
```

The training set is used to train the model.

The validation set is used during development for:

* monitoring model performance
* selecting the best checkpoint
* tuning prediction thresholds

The test set is used only for final evaluation.

The test set should not be used to choose the best model, tune thresholds, or compare too many experiments. Otherwise, it becomes another validation set.

---

## 3. Model Outputs

The model outputs raw logits.

These logits are converted to probabilities using sigmoid:

```python
probabilities = sigmoid(logits)
```

Then each probability is compared with a threshold:

```python
prediction = probability >= threshold
```

This gives the final positive or negative prediction for each label.

---

## 4. Why Threshold Tuning Is Used

A default threshold of `0.5` is not always good for imbalanced multi-label classification.

Some labels are frequent, while others are rare.
A rare label may need a lower or higher threshold than a frequent label.

For this reason, the project tunes one threshold per label using the validation set.

The threshold that gives the best F1-score for each label is selected.

These tuned thresholds are then used for:

* validation tuned metrics
* test tuned metrics
* API inference

The saved thresholds are stored in:

```text
thresholds.json
```

---

## 5. Metrics Used

The project reports both threshold-independent and threshold-dependent metrics.

### Threshold-independent metrics

These metrics evaluate ranking quality before choosing a final binary threshold:

```text
auc_macro
auc_micro
ap_macro
ap_micro
```

AUC and Average Precision are useful because they show whether the model ranks positive examples higher than negative examples.

### Threshold-dependent metrics

These metrics evaluate the final binary predictions after applying thresholds:

```text
f1_macro_tuned
f1_micro_tuned
precision_macro_tuned
recall_macro_tuned
```

These metrics are important because the API must finally return positive or negative predictions, not only probabilities.

---

## 6. Why Macro F1 Is Used for Best Checkpoint Selection

The best checkpoint is selected using:

```text
val/f1_macro_tuned
```

This is used because the dataset is imbalanced and rare labels matter.

Macro F1 calculates performance for each label and then averages the result.
This gives rare labels more importance than micro metrics.

Micro metrics are also useful, but they can be dominated by frequent labels.

---

## 7. Test Evaluation

Test evaluation can be enabled with:

```bash
python train.py train.run_test=true
```

The test set is evaluated using the best checkpoint.

During test evaluation, the project reports:

* global test metrics
* tuned test metrics
* per-label F1
* per-label precision
* per-label recall
* per-label thresholds

The test results should be used for final reporting, not for repeated model selection.
