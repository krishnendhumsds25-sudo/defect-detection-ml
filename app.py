import streamlit as st
import numpy as np
import cv2
import joblib
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Defect Detection",
    page_icon="🔍",
    layout="wide"
)

# ---------------------------
# Load Model
# ---------------------------
model = joblib.load("models/model.pkl")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("📌 About")
st.sidebar.info("Detect surface defects using Machine Learning")

st.sidebar.markdown("### Supported Defects:")
st.sidebar.write("""
- Crazing  
- Inclusion  
- Patches  
- Pitted Surface  
- Rolled-in Scale  
- Scratches  
""")

# ---------------------------
# Title
# ---------------------------
st.title("🔍 Surface Defect Detection System")
st.markdown("Upload an image to detect manufacturing defects")

# ---------------------------
# Upload Image
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is None:
    st.warning("⚠️ Please upload an image to continue")
    st.stop()

# ---------------------------
# Read Image
# ---------------------------
try:
    image = Image.open(uploaded_file).convert("L")
except:
    st.error("❌ Invalid image file")
    st.stop()

image = np.array(image)

# ---------------------------
# Preprocessing
# ---------------------------
image_resized = cv2.resize(image, (64, 64))
image_norm = image_resized / 255.0
flat = image_norm.flatten()

# ---------------------------
# Feature Engineering
# ---------------------------
mean = flat.mean()
std = flat.std()

edges = cv2.Canny(image_resized.astype("uint8"), 100, 200)
edge_feature = edges.mean()

# ---------------------------
# FIXED FEATURES (with names)
# ---------------------------
feature_array = np.concatenate([flat, [mean, std, edge_feature]])
columns = list(model.feature_names_in_)
features = pd.DataFrame([feature_array], columns=columns)

# ---------------------------
# Prediction
# ---------------------------
pred = model.predict(features)[0]

try:
    confidence = max(model.predict_proba(features)[0]) * 100
except:
    confidence = 0

# ---------------------------
# Layout
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🖼 Uploaded Image")
    st.image(image_resized, width=300)

    st.subheader("🧠 Edge Detection")
    st.image(edges, width=300)

with col2:
    st.subheader("📊 Prediction Result")

    st.success(f"Predicted Defect: {pred.upper()}")
    st.write(f"Confidence: {confidence:.2f}%")
    st.progress(int(confidence))

    # ✅ Added explanation line
    st.info("Prediction is based on patterns learned by the model from training data.")

    # ---------------------------
    # Features Display
    # ---------------------------
    st.subheader("📌 Extracted Features")

    col3, col4, col5 = st.columns(3)
    col3.metric("Mean", f"{mean:.3f}")
    col4.metric("Std Dev", f"{std:.3f}")
    col5.metric("Edge", f"{edge_feature:.2f}")

    # ---------------------------
    # Feature Chart
    # ---------------------------
    st.subheader("📊 Feature Visualization")

    fig, ax = plt.subplots()
    feature_names = ["Mean", "Std Dev", "Edge"]
    values = [mean, std, edge_feature]

    ax.bar(feature_names, values)
    ax.set_title("Feature Values")

    st.pyplot(fig)

    # ---------------------------
    # ✅ UPDATED MODEL INSIGHT
    # ---------------------------
    st.subheader("🧠 Model Insight")

    if pred == "inclusion":
        st.info("The model detected irregular internal patterns typical of inclusion defects.")
    elif pred == "patches":
        st.info("The model identified uneven surface regions indicating patch defects.")
    elif pred == "scratches":
        st.info("The model detected linear edge patterns consistent with scratches.")
    else:
        st.info("The prediction is based on learned feature patterns from the dataset.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Built using Machine Learning • Streamlit App")