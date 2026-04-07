import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Cross-Subject EEG Domain Adaptation",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------
# Static data
# ----------------------------
class_names = {
    0: "Left Hand",
    1: "Right Hand",
    2: "Feet",
    3: "Tongue"
}

model_results = pd.DataFrame({
    "Model": ["EEGNet", "ShallowConvNet", "DeepConvNet", "ATCNet", "Conformer"],
    "Accuracy": [0.56, 0.45, 0.43, 0.46, 0.29],
    "Weighted F1": [0.52, 0.45, 0.37, 0.42, 0.28]
})

coral_results = pd.DataFrame({
    "Model": ["EEGNet Baseline", "EEGNet + CORAL"],
    "Accuracy": [0.545, 0.569],
    "Weighted F1": [0.527, 0.510]
})

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Pipeline",
        "Model Comparison",
        "Domain Adaptation",
        "EEG Trial Demo"
    ]
)

# ----------------------------
# Helper functions
# ----------------------------
@st.cache_resource
def load_eegnet_model():
    model = load_model("models/eegnet_cross_subject_4class.keras")
    return model


def plot_model_results(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df["Model"]))
    width = 0.35

    ax.bar(x - width / 2, df["Accuracy"], width, label="Accuracy")
    ax.bar(x + width / 2, df["Weighted F1"], width, label="Weighted F1")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)


def plot_eeg_trial(trial: np.ndarray, max_channels: int = 8):
    fig, ax = plt.subplots(figsize=(12, 5))
    n_channels = min(trial.shape[0], max_channels)

    for ch in range(n_channels):
        ax.plot(trial[ch, :] + ch * 5, label=f"Ch {ch+1}")

    ax.set_title("Uploaded EEG Trial")
    ax.set_xlabel("Time Samples")
    ax.set_ylabel("Amplitude (offset)")
    st.pyplot(fig)


def get_confidence_label(max_prob: float) -> str:
    if max_prob >= 0.70:
        return "High confidence"
    elif max_prob >= 0.45:
        return "Moderate confidence"
    return "Low confidence"


def real_predict(trial: np.ndarray):
    """
    Predict one preprocessed EEG trial using the trained EEGNet model.

    Expected input shape:
    (25, 501)
    """
    model = load_eegnet_model()

    if trial.shape != (25, 501):
        raise ValueError(f"Expected shape (25, 501), but got {trial.shape}")

    # Add batch dimension and channel dimension
    trial_input = trial[np.newaxis, ..., np.newaxis]   # (1, 25, 501, 1)

    probs = model.predict(trial_input, verbose=0)[0]
    pred_class = int(np.argmax(probs))

    return pred_class, probs


# ----------------------------
# Pages
# ----------------------------
if page == "Overview":
    st.title("🧠 Cross-Subject EEG Motor Imagery Classification")
    st.subheader("with Domain Adaptation")

    st.write("""
    This project investigates **cross-subject EEG motor imagery classification**
    using deep learning and **CORAL domain adaptation**.

    The goal is to decode imagined movements from EEG signals across unseen subjects
    and study how domain adaptation can reduce inter-subject variability.
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Subjects", "9")
    c2.metric("Classes", "4")
    c3.metric("Best Model", "EEGNet")

    st.markdown("### Motor Imagery Classes")
    st.write("- Left Hand")
    st.write("- Right Hand")
    st.write("- Feet")
    st.write("- Tongue")

    st.markdown("### What This Project Covers")
    st.write("- EEG preprocessing and trial extraction")
    st.write("- Leave-One-Subject-Out (LOSO) cross-subject evaluation")
    st.write("- Comparison of multiple deep learning architectures")
    st.write("- CORAL-based domain adaptation")
    st.write("- Analysis of cross-subject generalization")

elif page == "Pipeline":
    st.title("⚙️ EEG Processing and Cross-Subject Pipeline")

    st.markdown("### Workflow")
    st.write("""
    **Raw EEG (GDF)** → **Bandpass Filtering** → **Event Extraction** →
    **Epoch Segmentation** → **Normalization** → **Cross-Subject LOSO** →
    **Deep Learning Models** → **Domain Adaptation**
    """)

    st.markdown("### Preprocessing Steps")
    st.write("- Bandpass filtering")
    st.write("- Event extraction from GDF annotations")
    st.write("- Epoch segmentation into trials")
    st.write("- Removal of EOG channels")
    st.write("- Normalization using training statistics only")

    st.markdown("### Cross-Subject Evaluation")
    st.write("""
    A **Leave-One-Subject-Out (LOSO)** setup was used:
    - train on 8 subjects
    - test on 1 unseen subject
    """)

    st.markdown("### Why This Matters")
    st.write("""
    Cross-subject decoding is more realistic than subject-specific classification,
    because real-world BCI systems must handle new users whose EEG signals differ
    from the training distribution.
    """)

elif page == "Model Comparison":
    st.title("📊 Model Comparison")

    st.markdown("### Single LOSO Split Results")
    st.dataframe(model_results, use_container_width=True)

    plot_model_results(model_results, "Model Performance on One LOSO Split")

    st.markdown("### Interpretation")
    st.write("""
    - **EEGNet** produced the strongest cross-subject baseline.
    - **ShallowConvNet** and **DeepConvNet** underperformed compared to EEGNet.
    - **ATCNet** did not surpass EEGNet despite greater architectural complexity.
    - The simplified **Conformer** performed worst in the current setup.
    """)

    st.write("""
    These results suggest that **EEG-specific inductive bias** is more valuable
    than simply increasing depth or adding attention for this task.
    """)

elif page == "Domain Adaptation":
    st.title("🔄 Domain Adaptation with CORAL")

    st.markdown("### What CORAL Does")
    st.write("""
    CORAL (Correlation Alignment) is a domain adaptation method that reduces
    distribution mismatch between source subjects and a target subject by aligning
    feature statistics.
    """)

    st.markdown("### Baseline vs CORAL-Adapted EEGNet")
    st.dataframe(coral_results, use_container_width=True)

    plot_model_results(coral_results, "EEGNet Baseline vs EEGNet + CORAL")

    st.markdown("### Interpretation")
    st.write("""
    - CORAL improved **overall accuracy**
    - Weighted F1-score dropped slightly
    - This suggests better global alignment, but slightly weaker class balance
    """)

    st.write("""
    This shows that simple domain adaptation can improve cross-subject generalization,
    though stronger methods may be needed to preserve class-level separability.
    """)

elif page == "EEG Trial Demo":
    st.title("🖥️ EEG Trial Demo")

    st.write("""
    Upload a **preprocessed EEG trial** saved as a `.npy` file.

    Expected shape:
    - `(25, 501)` for one trial
    """)

    with st.expander("Expected input format"):
        st.write("""
        The demo expects a single preprocessed EEG trial:
        - 25 EEG channels
        - 501 time samples
        - saved as a NumPy `.npy` file
        """)

    uploaded_file = st.file_uploader("Upload EEG Trial (.npy)", type=["npy"])

    if uploaded_file is not None:
        try:
            trial = np.load(uploaded_file)

            st.write("### Uploaded Trial Shape")
            st.code(str(trial.shape))

            if trial.ndim != 2:
                st.error("Expected a 2D EEG trial with shape (25, 501).")
            elif trial.shape != (25, 501):
                st.error(f"Expected shape (25, 501), but got {trial.shape}")
            else:
                st.success("Valid EEG trial loaded.")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write("### EEG Signal Visualization")
                    plot_eeg_trial(trial)

                with col2:
                    st.write("### Model Prediction")
                    pred_class, probs = real_predict(trial)

                    max_prob = float(np.max(probs))
                    confidence_label = get_confidence_label(max_prob)

                    st.metric("Predicted Class", class_names[pred_class])
                    st.metric("Top Probability", f"{max_prob:.3f}")
                    st.info(confidence_label)

                st.write("### Prediction Probabilities")
                probs_df = pd.DataFrame({
                    "Class": [class_names[i] for i in range(4)],
                    "Probability": probs
                })

                st.dataframe(probs_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(probs_df["Class"], probs_df["Probability"])
                ax.set_ylim(0, 1)
                ax.set_ylabel("Probability")
                ax.set_title("Prediction Confidence")
                st.pyplot(fig)

                st.write("### Interpretation")
                st.write("""
                The model outputs class probabilities across the four motor imagery classes.
                Lower confidence scores are expected in cross-subject EEG classification
                because brain signals vary significantly across individuals.
                """)

        except Exception as e:
            st.error(f"Could not process uploaded EEG trial: {e}")