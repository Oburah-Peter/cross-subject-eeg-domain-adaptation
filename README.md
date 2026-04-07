# 🧠 Cross-Subject EEG Motor Imagery Classification with Domain Adaptation

This project investigates how deep learning models perform on EEG-based motor imagery classification under **cross-subject conditions**, and explores how **domain adaptation** can improve generalization to unseen individuals.

The goal is to decode imagined movements (**Left Hand, Right Hand, Feet, Tongue**) from EEG signals and understand how models behave when applied to new subjects.

---

## 📸 Demo Preview

<p align="center">
  <img src="https://raw.githubusercontent.com/Oburah-Peter/cross-subject-eeg-domain-adaptation/main/demo.png" width="800"/>
</p>

---

## 🚀 Project Overview

EEG-based motor imagery classification is a fundamental task in **Brain-Computer Interfaces (BCIs)**. However, EEG signals vary significantly across individuals, making it challenging for models trained on one group of subjects to generalize to new users.

This project focuses on:

* Cross-subject EEG classification using deep learning
* Evaluating generalization with **Leave-One-Subject-Out (LOSO)**
* Applying **domain adaptation (CORAL)** to reduce inter-subject variability

---

## 🧪 Dataset

* **BCI Competition IV Dataset 2a**
* 9 subjects
* 4 motor imagery classes:

  * Left Hand
  * Right Hand
  * Feet
  * Tongue

---

## ⚙️ Pipeline

### 1. EEG Signal Processing

Raw EEG signals are transformed into model-ready trials:

* Bandpass filtering
* Event extraction from annotations
* Epoch segmentation
* Removal of non-relevant channels (e.g., EOG)
* Normalization using training statistics

---

### 2. Cross-Subject Evaluation (LOSO)

A **Leave-One-Subject-Out (LOSO)** setup is used:

* Train on 8 subjects
* Test on 1 unseen subject

This simulates real-world BCI deployment where models must generalize to new users.

---

### 3. Deep Learning Models

The following architectures were evaluated:

* **EEGNet (EEG-specific CNN)** ✅
* ShallowConvNet
* DeepConvNet
* ATCNet (CNN + Attention)
* EEG Conformer (CNN + Transformer-based attention)

---

## 📊 Model Performance (Single LOSO Split)

| Model          | Accuracy   | Weighted F1 |
| -------------- | ---------- | ----------- |
| EEGNet         | ~0.55–0.62 | ~0.52–0.62  |
| ShallowConvNet | ~0.45      | ~0.45       |
| DeepConvNet    | ~0.43      | ~0.37       |
| ATCNet         | ~0.46      | ~0.42       |
| Conformer      | ~0.29      | ~0.28       |

### 🧠 Key Insight

Models specifically designed for EEG (e.g., **EEGNet**) outperform deeper or more complex architectures in cross-subject settings.

Increasing architectural complexity does not necessarily improve performance on EEG data.

---

## 🔄 Domain Adaptation (CORAL)

To address inter-subject variability, this project applies **CORAL (Correlation Alignment)**.

### Concept

Domain adaptation reduces differences between:

* **Source domain** (training subjects)
* **Target domain** (unseen subject)

CORAL aligns feature distributions between these domains.

---

### Implementation

* Channel-wise CORAL (efficient approximation)
* Alignment applied to training data only
* No target labels used (unsupervised adaptation)

---

## 📈 Domain Adaptation Results

| Model           | Accuracy    | Weighted F1 |
| --------------- | ----------- | ----------- |
| EEGNet Baseline | 0.545       | 0.527       |
| EEGNet + CORAL  | **0.569 ↑** | 0.510       |

### 🧠 Domain Adaptation Insight

* CORAL improves overall accuracy
* Slight drop in F1 indicates reduced class balance
* Domain adaptation reduces inter-subject variability but may affect class separability

---

## 🖥️ Streamlit App

This project includes an interactive app for EEG prediction.

### ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 📥 Input Format

Upload a `.npy` file containing one EEG trial:

```
Shape: (25, 501)
```

A sample file is provided:

```
sample_trial.npy
```

---

## 🧠 Model Usage

The deployed app uses:

> ✅ Baseline EEGNet (cross-subject model)
> ❌ Domain adaptation is NOT applied during inference

---

## 📦 Model File

The trained model is located in:

```
models/eegnet_cross_subject_4class.keras
```

If missing, place the model manually in the `models/` directory.

---

## 🧪 Notebook

Full pipeline and experiments:

```
EEG_Classification.ipynb
```

---

## 🧩 Key Contributions

* End-to-end EEG classification pipeline
* Cross-subject LOSO evaluation
* Multi-architecture comparison
* Domain adaptation implementation (CORAL)
* Analysis of generalization vs class balance

---

## 🛠 Tech Stack

* Python
* NumPy, Pandas
* Scikit-learn
* TensorFlow / Keras
* MNE
* Matplotlib
* Streamlit

---

## 🚀 Future Work

* Fine-tuning with target-subject data
* Feature-level domain adaptation
* Domain adversarial networks (DANN)
* Real-time EEG systems
* FastAPI deployment for raw EEG (EDF/GDF)

---

## 🌍 Applications

* Brain-Computer Interfaces (BCIs)
* Assistive technologies
* Neuroadaptive systems
* Brain-controlled prosthetics
* Computational neuroscience

---

## 👤 Author

**Peter Otieno**
AI Engineer | Data Scientist | Machine Learning Researcher

---

⭐ Exploring intelligent systems at the intersection of AI and neuroscience.
