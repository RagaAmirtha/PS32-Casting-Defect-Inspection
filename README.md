# PS32-Casting-Defect-Inspection
Deep learning based casting defect inspection using MobileNetV2 transfer learning.

## 📌 Project Overview

This project addresses **casting defect inspection in submersible pump impellers** using deep learning-based image classification.

The objective is to classify casting images into two classes:

- **Defective**
- **OK**

In industrial inspection, missing a defective casting can be more costly than incorrectly rejecting a good casting. Therefore, this project focuses on achieving a **high recall for defective castings** while maintaining good overall classification performance.

---

## 🎯 Problem Statement

**PS-32: Casting Defect Inspection**

**Domain:** Industry & Infrastructure

Submersible pump impeller castings need to be inspected for defects. Manual inspection can be time-consuming and may result in missed defects.

The task is to develop a CNN-based image classification system that automatically identifies whether a casting is **Defective** or **OK**.

The evaluation particularly emphasizes **defective-class recall**, since false negatives can have a higher industrial cost than false positives.

---

## 📊 Dataset

The dataset contains approximately **7,300 grayscale casting images**, with an image size of **300 × 300 pixels**.

### Dataset used in this project

| Dataset | Number of Images |
|---|---:|
| Training images | 6,633 |
| Test images | 715 |
| **Total** | **7,348** |

### Classes

| Class | Training | Test |
|---|---:|---:|
| Defective | 3,758 | 453 |
| OK | 2,875 | 262 |

The training data was further divided into:

- **85% training**
- **15% validation**

The split was performed using a fixed random seed and stratification to maintain the class distribution.

> The original dataset is not included in this repository because of its size. Please obtain the dataset from the source specified in the project documentation/notebook.

---

## 🧠 Proposed Solution

A **MobileNetV2** architecture pretrained on ImageNet was used as the feature extractor.

MobileNetV2 was selected because it provides a good balance between:

- Classification performance
- Computational efficiency
- Model size
- Suitability for practical deployment

### Model Pipeline

```text
Casting Image
      ↓
Resize to 160 × 160
      ↓
Data Augmentation
      ↓
MobileNetV2
      ↓
Global Average Pooling
      ↓
Dropout
      ↓
Sigmoid Output
      ↓
Defective / OK
```

### Preprocessing

- Images resized from **300 × 300 → 160 × 160**
- Pixel normalization
- Data augmentation during training

### Data Augmentation

The following augmentation techniques were used:

- Random rotation
- Random zoom
- Random contrast adjustment

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Model | MobileNetV2 |
| Input size | 160 × 160 |
| Pretrained weights | ImageNet |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Binary Crossentropy |
| Dropout | 0.30 |
| Maximum epochs | 8 |
| Early stopping | Enabled |
| Learning-rate reduction | Enabled |

Class weights were also used during training to account for the difference in class frequencies.

---

## 🎚️ Decision Threshold

The default binary classification threshold of **0.50** was not used as the final operating point.

Since missing a defective casting is more costly than producing a false alarm, different thresholds were evaluated on the validation set.

A threshold of:

**0.30**

was selected because it provided a stronger defective-class recall and the highest F2 score among the evaluated thresholds.

This resulted in:

- **Defective Recall: 99.78%**
- False Negatives: **1**

This threshold prioritizes detecting defective castings, even at the cost of increasing false alarms on some OK castings.

---

## 📈 Results

The final model was evaluated on the unseen test set using a threshold of **0.30**.

### Overall Performance

| Metric | Result |
|---|---:|
| Test Accuracy | **96.92%** |
| Defective Precision | **95.56%** |
| Defective Recall | **99.78%** |
| Defective F1-score | **97.62%** |
| Macro F1-score | **96.63%** |

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| OK | 99.59% | 91.98% | 95.63% |
| Defective | 95.56% | **99.78%** | 97.62% |
| **Macro Average** | **97.58%** | **95.88%** | **96.63%** |

---

## 🔲 Confusion Matrix

The final test confusion matrix was:

| Actual / Predicted | OK | Defective |
|---|---:|---:|
| **OK** | 241 | 21 |
| **Defective** | 1 | 452 |

Therefore:

- **452 defective castings** were correctly detected.
- Only **1 defective casting** was missed.
- **241 OK castings** were correctly classified.
- **21 OK castings** were incorrectly flagged as defective.

The model therefore strongly prioritizes the detection of defective castings.

---

## 🔍 Misclassification Analysis

There were **22 misclassified test images** in total:

- **1 False Negative**
- **21 False Positives**

The false-negative image was:

```text
cast_def_0_150.jpeg
```

with a predicted defective probability of approximately:

```text
0.2778
```

Since the selected threshold was 0.30, this image was classified as OK even though it was actually defective.

Several OK images were classified as defective because their visual patterns were sufficiently similar to defective castings.

This demonstrates the trade-off introduced by lowering the decision threshold: **defective recall increases, but false positives also increase.**

---

## 💾 Trained Model

The trained model is saved as:

```text
casting_defect_mobilenetv2.keras
```

This file contains the trained neural-network architecture and learned weights and can be used for future inference without retraining the model.

---

## 📁 Repository Structure

```text
PS32-Casting-Defect-Inspection/
│
├── PS32_Casting_Defect_Inspection.ipynb
├── casting_defect_mobilenetv2.keras
├── requirements.txt
├── PS32_Casting_Defect_Inspection_Report.pdf
└── README.md
```

The original dataset is intentionally not included because of its size.

---

## 🛠️ Installation

Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
tensorflow
numpy
pandas
matplotlib
scikit-learn
seaborn
pillow
```

---

## ▶️ Running the Project

1. Download or obtain the casting defect dataset.
2. Arrange the dataset according to the folder structure expected by the notebook.
3. Open:

```text
PS32_Casting_Defect_Inspection.ipynb
```

4. Install the required dependencies.
5. Run the notebook cells sequentially.

The notebook performs:

```text
Dataset Loading
      ↓
Class Distribution
      ↓
Image Visualization
      ↓
Train/Validation Split
      ↓
Preprocessing & Augmentation
      ↓
MobileNetV2 Training
      ↓
Validation
      ↓
Threshold Selection
      ↓
Test Evaluation
      ↓
Confusion Matrix
      ↓
Misclassification Analysis
```

---

## 🏭 Industrial Relevance

The proposed system can assist automated visual inspection of casting components.

The high defective-class recall is particularly important because a missed defective casting may result in:

- Product quality issues
- Additional inspection requirements
- Equipment failure risks
- Increased manufacturing costs

The system can therefore be used as a **decision-support or screening tool** to help inspectors identify potentially defective castings.

---

## ⚠️ Limitations

- The model performance depends on the quality and distribution of the training dataset.
- The system was evaluated on the provided casting-image dataset and may require additional validation on real production-line images.
- False positives remain possible, particularly after lowering the classification threshold.
- Lighting, camera conditions, casting orientation, and surface appearance may affect predictions.

---

## 🚀 Future Improvements

Possible future improvements include:

- Testing additional lightweight CNN architectures
- Quantization for edge deployment
- Testing on real production-line images
- Increasing dataset diversity
- Improving defect localization using object detection or segmentation
- Evaluating the model under different lighting and imaging conditions
- Optimizing inference speed for industrial deployment

---

## 📚 References

1. Dataset: Casting defect image dataset provided for the PS-32 Casting Defect Inspection challenge.
2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. **MobileNetV2: Inverted Residuals and Linear Bottlenecks.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018.
3. TensorFlow/Keras documentation for MobileNetV2 and image classification.

---

## 🤖 AI Usage Declaration

AI tools were used as an assistance tool during the development of this project for:

- Understanding implementation steps
- Debugging programming issues
- Improving code readability
- Structuring documentation and explanations

The dataset processing, model training, evaluation, and reported experimental results were performed as part of the project workflow.
