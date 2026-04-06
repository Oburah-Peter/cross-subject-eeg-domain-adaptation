import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Cross-Subject EEG Domain Adaptation",
    page_icon="🧠",
    layout="wide"
)

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
# Static project data
# ----------------------------
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

class_names = {
    0: "Left Hand",
    1: "Right Hand",
    2: "Feet",
    3: "Tongue"
}

# ----------------------------
# Helper functions
# ----------------------------
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


def dummy_predict(trial: np.ndarray):
    """
    Placeholder prediction function.
    Replace this with your real EEGNet model inference later.
    """
    rng = np.random.default_rng(42)
    probs = rng.random(4)
    probs = probs / probs.sum()
    pred = int(np.argmax(probs))
    return pred, probs


# ----------------------------
# Pages
# ----------------------------
if page == "Overview":
    st.title("🧠 Cross-Subject EEG Motor Imagery Classification")
    st.subheader("with Domain Adaptation")

    st.write("""
    This project investigates **cross-subject EEG motor imagery classification**
    using deep learning models and **CORAL domain adaptation**.

    The goal is to decode imagined movements from EEG signals across unseen subjects
    and study how domain adaptation can reduce inter-subject variability.
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Subjects", "9")
    c2.metric("Classes", "4")
    c3.metric("Best Model So Far", "EEGNet")

    st.markdown("### Motor Imagery Classes")
    st.write("- Left Hand")
    st.write("- Right Hand")
    st.write("- Feet")
    st.write("- Tongue")

    st.markdown("### Key Contributions")
    st.write("- Built an end-to-end EEG preprocessing pipeline")
    st.write("- Implemented Leave-One-Subject-Out (LOSO) evaluation")
    st.write("- Compared multiple deep learning architectures")
    st.write("- Applied CORAL for domain adaptation")
    st.write("- Analyzed trade-offs between accuracy and class balance")

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
    - repeat across subjects
    """)

    st.markdown("### Why This Matters")
    st.write("""
    EEG signals vary across individuals, making cross-subject decoding difficult.
    This setup reflects a more realistic BCI scenario than subject-specific testing.
    """)

elif page == "Model Comparison":
    st.title("📊 Model Comparison")

    st.markdown("### Single LOSO Split Results")
    st.dataframe(model_results, use_container_width=True)

    plot_model_results(model_results, "Model Performance on One LOSO Split")

    st.markdown("### Interpretation")
    st.write("""
    - **EEGNet** performed best and provided the strongest cross-subject baseline.
    - **ShallowConvNet** and **DeepConvNet** underperformed compared to EEGNet.
    - **ATCNet** did not surpass EEGNet despite higher architectural complexity.
    - The simplified **Conformer** performed worst in the current setup.
    """)

    st.write("""
    These results suggest that **EEG-specific architectures** are better suited
    for cross-subject motor imagery classification than generic deeper or attention-heavy models.
    """)

elif page == "Domain Adaptation":
    st.title("🔄 Domain Adaptation with CORAL")

    st.markdown("### What CORAL Does")
    st.write("""
    CORAL (Correlation Alignment) reduces domain shift by aligning
    the feature distribution of the training data with that of the target subject.
    """)

    st.markdown("### Baseline vs CORAL-Adapted EEGNet")
    st.dataframe(coral_results, use_container_width=True)

    plot_model_results(coral_results, "EEGNet Baseline vs EEGNet + CORAL")

    st.markdown("### Interpretation")
    st.write("""
    - Accuracy improved after CORAL adaptation.
    - Weighted F1-score dropped slightly.
    - This suggests that **global alignment helped overall prediction performance**,
      but class-level balance became slightly worse.
    """)

    st.write("""
    This shows that simple domain adaptation can reduce inter-subject variability,
    but stronger methods may be needed to preserve class separability.
    """)

elif page == "EEG Trial Demo":
    st.title("🖥️ EEG Trial Demo")

    st.write("""
    Upload a **preprocessed EEG trial** saved as a `.npy` file.

    Expected shape:
    - `(25, 501)` for one trial
    """)

    uploaded_file = st.file_uploader("Upload EEG Trial (.npy)", type=["npy"])

    if uploaded_file is not None:
        try:
            trial = np.load(uploaded_file)

            st.write("### Uploaded Trial Shape")
            st.code(str(trial.shape))

            if trial.ndim != 2:
                st.error("Expected a 2D trial with shape (channels, time_samples).")
            else:
                st.success("Valid EEG trial loaded.")

                st.write("### EEG Signal Visualization")
                plot_eeg_trial(trial)

                st.write("### Demo Prediction")
                pred_class, probs = dummy_predict(trial)

                st.metric("Predicted Class", class_names[pred_class])

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

                st.info("This demo currently uses a placeholder predictor. Replace `dummy_predict()` with your trained EEGNet inference pipeline.")
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")