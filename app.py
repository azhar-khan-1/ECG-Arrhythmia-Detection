# ============================================================
# ECG Arrhythmia Detection using Deep Learning
# Developed by Azhar Khan | IIT Indore
# ============================================================

import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="ECG Arrhythmia Detection",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------

st.markdown("""
<style>

.main-title{
    font-size:48px;
    font-weight:bold;
    color:#00E5FF;
}

.sub-title{
    font-size:22px;
    color:#CFCFCF;
}

.metric-card{
    padding:15px;
    border-radius:12px;
    background:#20232A;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "models/ecg_arrhythmia_cnn.keras"
    )

    return model

model = load_model()

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

@st.cache_data
def load_dataset():

    X_test = np.load("dataset/X_test.npy")
    y_test = np.load("dataset/y_test.npy")

    return X_test, y_test

X_test, y_test = load_dataset()

# ------------------------------------------------------------
# Class Names
# ------------------------------------------------------------

class_names = {

    0: "Atrial Beat (A)",
    1: "Normal Beat (N)",
    2: "Ventricular Beat (V)"

}

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.title("🫀 ECG Arrhythmia Detection")

st.sidebar.markdown("---")

st.sidebar.markdown("""
### 📌 Model

**1D Convolutional Neural Network**

- Conv1D
- Batch Normalization
- MaxPooling
- Dropout
- Global Average Pooling

""")

st.sidebar.markdown("""
### 📊 Dataset

MIT-BIH Arrhythmia Database

Classes

- 🟢 Normal
- 🟠 Atrial
- 🔴 Ventricular

""")

st.sidebar.markdown("""
### 📈 Performance

✅ Validation Accuracy

**96.75%**

✅ ROC-AUC

**99.26%**

""")

st.sidebar.markdown("---")

sample_index = st.sidebar.slider(

    "Select ECG Sample",

    0,

    len(X_test)-1,

    100

)

st.sidebar.markdown("---")

st.sidebar.info(
    "Developed by\n\nAzhar Khan\n\nM.Tech Biomedical Engineering\nIIT Indore"
)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

sample = X_test[sample_index]

actual_label = y_test[sample_index]

prediction = model.predict(

    sample.reshape(1,200,1),

    verbose=0

)

predicted_class = np.argmax(prediction)

confidence = np.max(prediction)

# ------------------------------------------------------------
# Title
# ------------------------------------------------------------

st.markdown(
    '<p class="main-title">🫀 ECG Arrhythmia Detection using Deep Learning</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Automated ECG Heartbeat Classification using a 1D CNN trained on the MIT-BIH Arrhythmia Database.</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ------------------------------------------------------------
# Main Dashboard
# ------------------------------------------------------------

col1, col2 = st.columns([2.2, 1])

# ------------------------------------------------------------
# ECG Plot
# ------------------------------------------------------------

with col1:

    st.subheader("📈 ECG Heartbeat Signal")

    fig, ax = plt.subplots(figsize=(11,4))

    ax.plot(
        sample,
        color="#00E5FF",
        linewidth=2
    )

    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Normalized Amplitude")

    ax.grid(alpha=0.3)

    st.pyplot(fig)

# ------------------------------------------------------------
# Prediction Card
# ------------------------------------------------------------

with col2:

    st.subheader("🤖 AI Prediction")

    if predicted_class == actual_label:

        st.success(class_names[predicted_class])

        st.success("✅ Prediction Correct")

    else:

        st.error(class_names[predicted_class])

        st.error("❌ Prediction Incorrect")

    st.metric(

        "Confidence",

        f"{confidence*100:.2f}%"

    )

    st.metric(

        "Actual Class",

        class_names[actual_label]

    )

    st.metric(

        "Predicted Class",

        class_names[predicted_class]

    )

# ------------------------------------------------------------
# Probability Chart
# ------------------------------------------------------------

st.markdown("---")

st.subheader("📊 Prediction Probabilities")

labels = [
    "Atrial",
    "Normal",
    "Ventricular"
]

colors = [
    "#ff9800",
    "#4CAF50",
    "#F44336"
]

fig2, ax2 = plt.subplots(figsize=(8,4))

bars = ax2.bar(

    labels,

    prediction[0],

    color=colors

)

ax2.set_ylim(0,1)

ax2.set_ylabel("Probability")

ax2.set_title("Class Probability Distribution")

for bar in bars:

    h = bar.get_height()

    ax2.text(

        bar.get_x()+bar.get_width()/2,

        h+0.02,

        f"{h*100:.1f}%",

        ha="center",

        fontsize=11,

        fontweight="bold"

    )

st.pyplot(fig2)

# ------------------------------------------------------------
# Model Performance
# ------------------------------------------------------------

st.markdown("---")

st.subheader("📈 Model Performance")

m1,m2,m3,m4 = st.columns(4)

m1.metric(
    "Validation Accuracy",
    "96.75%"
)

m2.metric(
    "ROC-AUC",
    "99.26%"
)

m3.metric(
    "Classes",
    "3"
)

m4.metric(
    "Parameters",
    "158K"
)

# ------------------------------------------------------------
# Model Information
# ------------------------------------------------------------

st.markdown("---")

with st.expander("🧠 CNN Architecture"):

    st.markdown("""

### Model Architecture

Input (200 × 1)

↓

Conv1D (64 Filters)

↓

Batch Normalization

↓

MaxPooling

↓

Dropout

↓

Conv1D (128 Filters)

↓

Batch Normalization

↓

MaxPooling

↓

Dropout

↓

Conv1D (256 Filters)

↓

Global Average Pooling

↓

Dense (128)

↓

Softmax (3 Classes)

""")

with st.expander("📚 Dataset Information"):

    st.markdown("""

**Dataset**

MIT-BIH Arrhythmia Database

**Heartbeat Classes**

- 🟢 Normal Beat (N)

- 🟠 Atrial Beat (A)

- 🔴 Ventricular Beat (V)

Total ECG Beats

**82,408**

""")

with st.expander("⚠ Medical Disclaimer"):

    st.warning("""

This application has been developed for

educational and research purposes only.

It must not be used for real-world medical diagnosis.

Always consult a qualified healthcare professional.

""")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.markdown("---")

st.markdown("""

<div style='text-align:center;'>

### ❤️ ECG Arrhythmia Detection using Deep Learning

Developed by **Azhar Khan**

M.Tech Biomedical Engineering

**Indian Institute of Technology Indore**

TensorFlow • Streamlit • Deep Learning • Medical AI

© 2026 All Rights Reserved

</div>

""", unsafe_allow_html=True)

