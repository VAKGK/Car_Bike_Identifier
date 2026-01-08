import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import os  # <--- Added this to handle file paths

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="Car and Bike Classifier",
    page_icon="🚗",
    layout="wide"
)

# ==================================================
# Color Theme
# ==================================================
COLOR_BG = "#fffdf7"
COLOR_TERRACOTTA = "#a67c5b"
COLOR_CHARCOAL = "#3d3d3d"
COLOR_TEXT = "#3d3d3d"

# ==================================================
# Custom CSS Styling
# ==================================================
st.markdown(f"""
<style>
.stApp {{
    background-color: {COLOR_BG};
}}

.big-font {{
    font-size: 32px;
    font-weight: bold;
    text-align: center;
    color: {COLOR_TEXT};
}}

.subtitle {{
    text-align: center;
    font-size: 16px;
    margin-bottom: 30px;
}}

.result-box {{
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    margin-top: 25px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}}

.car-box {{
    background: linear-gradient(135deg, #f5e9d9, #e8d4b8);
    border-left: 10px solid {COLOR_TERRACOTTA};
}}

.bike-box {{
    background: linear-gradient(135deg, #f0f0f0, #e0e0e0);
    border-left: 10px solid {COLOR_CHARCOAL};
}}

.stProgress > div > div > div {{
    background-color: {COLOR_TERRACOTTA};
}}

.bike-progress .stProgress > div > div > div {{
    background-color: {COLOR_CHARCOAL};
}}
</style>
""", unsafe_allow_html=True)

# ==================================================
# Header
# ==================================================
st.markdown('<div class="big-font">Car and Bike Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image and let the AI decide</div>', unsafe_allow_html=True)
st.info("📌 Tip: Clear images with a single vehicle work best.")

# ==================================================
# Load TFLite Model (FIXED)
# ==================================================
@st.cache_resource
def load_tflite_model():
    # Get the path to the current file (app.py)
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # Create the absolute path to the model file
    model_path = os.path.join(curr_dir, "car_bike_model.tflite")
    
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==================================================
# Image Preprocessing
# ==================================================
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image)

    if image.shape[-1] == 4:  # RGBA → RGB
        image = image[:, :, :3]

    image = image / 255.0
    image = np.expand_dims(image, axis=0).astype("float32")
    return image

# ==================================================
# File Upload
# ==================================================
uploaded_file = st.file_uploader(
    "📤 Upload an image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🧠 Analyzing image..."):
        processed_image = preprocess_image(image)

        # Set input tensor
        interpreter.set_tensor(input_details[0]["index"], processed_image)
        interpreter.invoke()

        # Get output
        prediction = interpreter.get_tensor(output_details[0]["index"])
        pred = float(prediction[0][0])

    # ==================================================
    # Prediction Logic
    # ==================================================
    if pred > 0.80:
        label = "🚗 Car"
        confidence = pred
        box_class = "car-box"
        progress_class = ""
    elif pred < 0.20:
        label = "🏍️ Bike"
        confidence = 1 - pred
        box_class = "bike-box"
        progress_class = "bike-progress"
    else:
        label = "🤔 Not Sure"
        confidence = max(pred, 1 - pred)
        box_class = ""
        progress_class = ""

    # ==================================================
    # Result Display
    # ==================================================
    st.markdown(
        f"""
        <div class="result-box {box_class}">
            <h2>{label}</h2>
            <p style="font-size:18px;">Confidence: {confidence:.2%}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(f'<div class="{progress_class}">', unsafe_allow_html=True)
    st.progress(confidence)
    st.markdown("</div>", unsafe_allow_html=True)
