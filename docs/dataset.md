# Dataset Documentation

This document describes the dataset used in this project, why it was selected, how it was processed, and the main limitations that should be considered when interpreting the model results.

---

## 1. Dataset Source

This project uses the **PadChest** dataset, specifically a resized 224x224 version prepared for use with TorchXRayVision.

PadChest is a large public chest X-ray dataset collected at Hospital San Juan Hospital in Spain between 2009 and 2017. It contains more than 160,000 chest X-ray images from around 67,000 patients. The dataset includes multiple projection views, patient demographic information, image acquisition metadata, and labels extracted from radiology reports.

The reports were annotated with:

* 174 radiographic findings
* 19 differential diagnoses
* 104 anatomical locations

The labels are organized using a hierarchical taxonomy and mapped to UMLS medical terminology.

Dataset reference:

```bibtex
@article{padchest_sj_resized_224,
  title    = {PADCHEST_SJ (Resized 224x224) (Fixed cropping)},
  keywords = {chest xray},
  author   = {},
  abstract = {For use here: https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py#L472

Images are resized to 224x224 from the original dataset.

This dataset includes more than 160,000 images obtained from 67,000 patients that were interpreted and reported by radiologists at Hospital San Juan Hospital (Spain) from 2009 to 2017, covering six different position views and additional information on image acquisition and patient demography. The reports were labeled with 174 different radiographic findings, 19 differential diagnoses and 104 anatomic locations organized as a hierarchical taxonomy and mapped onto standard Unified Medical Language System (UMLS) terminology.

https://i.imgur.com/MpVlYgB.png

Padchest},
  terms    = {},
  license  = {Creative Commons Attribution-ShareAlike 4.0 International License},
  superseded = {},
  url      = {https://arxiv.org/abs/1901.07441}
}
```

Original paper:

```text
https://arxiv.org/abs/1901.07441
```

---

## 2. Why PadChest Was Used

PadChest was selected because it is one of the largest publicly available chest X-ray datasets with many lung-related findings and multi-label annotations.

The main goal of this project is to build and test a chest X-ray pathology classification pipeline that may later be evaluated on a small test dataset from a Moroccan hospital. Since PadChest was collected from a hospital in Spain, it was considered a reasonable public dataset for early experimentation because the population and imaging context may be closer to Moroccan clinical data than some other public datasets.

For this project, PadChest is used as a research dataset to:

* Train multi-label chest X-ray classification models
* Test different model architectures and loss functions
* Study the effect of projection selection
* Evaluate threshold tuning for imbalanced labels
* Build a complete training and deployment pipeline before testing on local clinical data

---

## 3. Raw PadChest Overview

The raw PadChest dataset contains chest X-ray studies and their associated radiology reports.

Important characteristics of the dataset include:

* Large-scale chest X-ray image collection
* Multiple projection views, including PA, AP, lateral, and other views
* Patient-level metadata
* DICOM acquisition metadata
* Labels extracted from radiology reports
* A mix of manually annotated and automatically labeled reports
* Hierarchical label organization for findings, diagnoses, and anatomical locations

The dataset is not a simple single-label classification dataset. Each image or study may have multiple labels, and the labels come from the radiology report associated with the study.

This is important because a study can contain more than one image, for example PA and lateral views, while sharing the same report labels.

---

## 4. Task Definition

The task in this project is:

```text
Multi-label lung pathology classification from chest X-ray images.
```

Given a chest X-ray image, the model predicts the probability of each selected pathology or radiographic finding.

This is a multi-label classification problem because one image can contain:

* No positive pathology label
* One positive label
* Multiple positive labels at the same time

The model outputs one probability per label using a sigmoid activation. A label is considered positive when its predicted probability is greater than or equal to its decision threshold.

---

## 5. Labels Used in This Project

The final labels used by the model are determined during preprocessing.

The preprocessing pipeline:

* Reads the PadChest labels
* Selects the target label set
* Optionally applies label normalization or aliases
* Filters labels that are not part of the selected target set
* Removes rare labels below a minimum count
* Saves the final label list to `labels_used.txt`

The final label file is stored inside the generated data mart:

```text
labels_used.txt
```

This file is used by the training pipeline to define the number and order of model outputs.

Because labels can change depending on preprocessing options, the exact label list should always be taken from the corresponding `labels_used.txt` file used during training.

---

## 6. Projection / View Selection

Different data marts can be created using different projection views.

The project supports experiments with:

* PA only
* PA and AP
* PA and lateral
* PA, lateral, and AP
* All available projections

The PA projection contains the majority of images and is often used as a cleaner baseline because it is one of the standard chest X-ray views.

Projection selection is important because different views can affect the appearance of anatomical structures and pathologies. For example, AP and PA views can make some findings appear differently. If projection types are not controlled, the model may learn projection-related bias instead of pathology-related patterns.

---

## 7. Train / Validation / Test Split

The dataset is split into:

```text
70% training
10% validation
20% test
```

The split is performed at the patient level to reduce data leakage.

This means images from the same patient should not appear in more than one split. This is important because the same patient can have multiple images or multiple studies. If the same patient appears in both training and testing, the test results may become too optimistic.

The splitting strategy also uses rare-label-aware stratification. The goal is to keep the label distribution as balanced as possible across train, validation, and test sets, especially for rare labels.

The split strategy is designed to:

* Avoid patient leakage
* Keep similar label distributions across splits
* Preserve rare labels as much as possible
* Produce more reliable validation and test evaluation

---

## 8. Generated Data Mart Structure

After preprocessing, the output data mart has the following structure:

```text
data_marts/<data_mart_name>/
├── images/
├── train.csv
├── val.csv
├── test.csv
├── labels_used.txt
├── label_map.csv
├── split_summary.csv
├── label_distribution_by_split.csv
├── config.json
└── <data_mart_name>_full.csv
```

Main files:

| File                              | Description                                              |
| --------------------------------- | -------------------------------------------------------- |
| `images/`                         | Preprocessed or copied image files                       |
| `train.csv`                       | Training split                                           |
| `val.csv`                         | Validation split                                         |
| `test.csv`                        | Test split                                               |
| `labels_used.txt`                 | Final label columns used for training                    |
| `label_map.csv`                   | Mapping between original labels and safe column names    |
| `split_summary.csv`               | Number of images, patients, and studies in each split    |
| `label_distribution_by_split.csv` | Per-label distribution for train, validation, and test   |
| `config.json`                     | Preprocessing configuration used to create the data mart |
| `<data_mart_name>_full.csv`       | Full filtered dataset before splitting                   |

The training pipeline expects each split CSV to contain:

* `image_path`
* patient/study metadata
* one binary column per label

---

## 9. Known Dataset Limitations

PadChest is a valuable dataset, but it has important limitations.

### Report-derived labels

The labels are extracted from radiology reports. This means they are weak image-level labels, not perfect ground-truth annotations drawn directly on the image.

A finding may be visible in the image but not mentioned in the report. Also, some reported findings may refer to the whole study rather than one specific image.

### Mixed manual and automatic labeling

Only part of the dataset was manually annotated by physicians. The remaining labels were generated using an automatic text-labeling model trained on the manually labeled reports.

This can introduce label noise.

### Multi-image studies

Some studies contain multiple images, such as PA and lateral views, but share the same report. As a result, different images from the same study can receive the same labels even if a finding is more visible in one view than another.

### Class imbalance

The dataset is highly imbalanced. Some findings are common, while others are very rare. This affects model training and evaluation, especially for macro F1-score and rare-label performance.

### Single-institution dataset

PadChest comes from one hospital system in Spain. Models trained on this dataset may not generalize well to images from other hospitals, countries, machines, or patient populations.

### Not clinically validated

The models trained in this project are experimental. They have not been prospectively evaluated in a real clinical setting.

---

## 10. Notes About Medical Use

This project was created as part of a Master’s final project and internship work.

The goal is to build and demonstrate a research pipeline for chest X-ray pathology classification, including preprocessing, model training, experiment tracking, model packaging, inference API deployment, and Grad-CAM visualization.

The trained models in this repository are not intended for medical diagnosis or clinical decision-making.

They should only be used for:

* Research
* Education
* Technical demonstration
* Pipeline testing
* Experimental evaluation

If the code is cloned and adapted for real clinical data, the resulting model would still require proper validation, clinical review, regulatory consideration, and testing on representative local data before any real medical use.

The current models should not be used to diagnose patients.
