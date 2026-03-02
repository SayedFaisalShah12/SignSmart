import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SignSmart - Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# --- CLASS NAMES MAPPING ---
CLASSES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

# --- STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        background-color: #0d6efd;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .result-text {
        font-size: 24px;
        font-weight: bold;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# --- APP LAYOUT ---
st.title("🚦 SignSmart: Traffic Sign Recognition")
st.markdown("Upload a traffic sign image to identify it using Deep Learning.")

with st.sidebar:
    st.header("Project Info")
    st.info("This app uses a Convolutional Neural Network (CNN) trained on the GTSRB dataset.")
    
    st.header("Model Selection")
    model_choice = st.selectbox("Choose Model Architecture", ["Custom CNN", "MobileNetV2"])
    
    # Path settings
    MODELS_DIR = "models"
    custom_model_path = os.path.join(MODELS_DIR, "custom.keras")
    pretrained_model_path = os.path.join(MODELS_DIR, "pretrained.keras")


# --- UTILS ---
@st.cache_resource
def load_prediction_model(model_name):
    path = custom_model_path if model_name == "Custom CNN" else pretrained_model_path
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)

def preprocess_image(image, model_name):
    # Resize based on model
    size = (30, 30) if model_name == "Custom CNN" else (224, 224)
    # Convert PIL to OpenCV format
    img_array = np.array(image.convert('RGB'))
    img_array = cv2.resize(img_array, size)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# --- MAIN INTERFACE ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

with col2:
    st.subheader("Classification Result")
    
    if uploaded_file is not None:
        model = load_prediction_model(model_choice)
        
        if model is None:
            st.error(f"Error: {model_choice} model file not found in '{MODELS_DIR}/'. Please train the model first.")
        else:
            # Preprocess
            processed_img = preprocess_image(image, model_choice)
            
            # Predict
            with st.spinner("Analyzing..."):
                prediction = model.predict(processed_img)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
            
            # Display results
            st.markdown(f"""
            <div class="prediction-card">
                <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">Predicted Class:</p>
                <p class="result-text">{CLASSES.get(class_id, "Unknown")}</p>
                <hr>
                <p>Confidence: <b>{confidence*100:.2f}%</b></p>
                <p>Class ID: <b>{class_id}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show additional details
            if confidence < 0.7:
                st.warning("Confidence is low. Results might be inaccurate.")
    else:
        st.write("Please upload an image to see results.")

# --- FOOTER ---
st.divider()
st.markdown("Developed with ❤️ using TensorFlow & Streamlit.")
