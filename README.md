# 🧠 Cross-Subject EEG Motor Imagery Classification with Domain Adaptation

This project investigates how deep learning models perform on EEG-based motor imagery classification under **cross-subject conditions**, and explores how **domain adaptation** can improve generalization to unseen individuals.

The goal is to decode imagined movements (Left Hand, Right Hand, Feet, Tongue) from EEG signals and understand how models behave when applied to new subjects.

---

## 🚀 Project Overview

EEG-based motor imagery classification is a core task in Brain-Computer Interfaces (BCIs). However, EEG signals vary significantly across individuals, making it difficult for models trained on one group of subjects to generalize to new users.

This project focuses on:

- Cross-subject EEG classification using deep learning
- Evaluating model generalization under **Leave-One-Subject-Out (LOSO)**
- Applying **domain adaptation (CORAL)** to reduce inter-subject variability

---

## 🧪 Dataset

- **BCI Competition IV Dataset 2a**
- 9 subjects
- 4 motor imagery classes:
  - Left Hand
  - Right Hand
  - Feet
  - Tongue

---

## ⚙️ Pipeline

### 1. EEG Signal Processing

The raw EEG signals are processed into model-ready trials:

- Bandpass filtering
- Event extraction from annotations
- Epoch segmentation into trials
- Removal of non-relevant channels (e.g., EOG)
- Normalization using training statistics

---

### 2. Cross-Subject Evaluation (LOSO)

A **Leave-One-Subject-Out (LOSO)** setup is used:

- Train on 8 subjects
- Test on 1 unseen subject

This simulates real-world BCI scenarios where models must generalize to new users.

---

### 3. Deep Learning Models

The following architectures were evaluated:

- **EEGNet** (EEG-specific CNN) ✅
- ShallowConvNet
- DeepConvNet
- ATCNet (CNN + Attention)
- EEG Conformer (CNN + Transformer-style attention)

---

## 📊 Model Performance (Single LOSO Split)

| Model | Accuracy | Weighted F1 |
|------|---------|------------|
| EEGNet | ~0.55–0.62 | ~0.52–0.62 |
| ShallowConvNet | ~0.45 | ~0.45 |
| DeepConvNet | ~0.43 | ~0.37 |
| ATCNet | ~0.46 | ~0.42 |
| Conformer | ~0.29 | ~0.28 |

---

## 🧠 Key Insight

Models specifically designed for EEG signals (e.g., EEGNet) perform better than deeper or more complex architectures under cross-subject conditions.

> Increasing model complexity does not necessarily improve EEG classification performance.

---

## 🔄 Domain Adaptation (CORAL)

To address inter-subject variability, this project applies **CORAL (Correlation Alignment)**.

### Concept

Domain adaptation methods aim to reduce differences between:
- **source domain** (training subjects)
- **target domain** (unseen subject)

CORAL aligns the statistical distribution of features between these domains. :contentReference[oaicite:0]{index=0}

---

### Implementation

- Channel-wise CORAL (efficient approximation)
- Alignment applied only to training data
- No target labels used (unsupervised adaptation)

---

## 📈 Domain Adaptation Results

| Model | Accuracy | Weighted F1 |
|------|---------|------------|
| EEGNet Baseline | 0.545 | 0.527 |
| EEGNet + CORAL | **0.569** | 0.510 |

---

## 🧠 Domain Adaptation Insight

- CORAL improves **overall accuracy**
- Slight drop in F1 indicates reduced class balance

> Domain adaptation helps reduce inter-subject variability, but simple feature alignment may not fully preserve class separability.

---

## 🧩 Key Contributions

- Built an **end-to-end EEG classification pipeline**
- Implemented **cross-subject (LOSO) evaluation**
- Compared multiple deep learning architectures
- Applied **domain adaptation (CORAL)**
- Analyzed trade-offs between generalization and class balance

---

## 🛠 Tech Stack

- Python
- NumPy, Pandas
- Scikit-learn
- TensorFlow / Keras
- MNE (EEG processing)
- Matplotlib

---

## 🚀 Future Work

- Fine-tuning with small target-subject data
- Feature-level domain adaptation
- Domain adversarial networks (DANN)
- Real-time EEG inference systems
- FastAPI-based deployment for raw EEG (EDF/GDF)

---

## 🌍 Applications

- Brain-Computer Interfaces (BCIs)
- Assistive technologies
- Neuroadaptive systems
- Brain-controlled prosthetics
- Computational neuroscience research

---

## 🔗 Repository

👉 https://github.com/Oburah-Peter/cross-subject-eeg-domain-adaptation

---

## 👤 Author

**Peter Otieno**  
AI Engineer | Data Scientist | Machine Learning Researcher  

---

⭐ Exploring intelligent systems at the intersection of AI and neuroscience.