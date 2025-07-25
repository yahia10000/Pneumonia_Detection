import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from PIL import Image
import streamlit as st
import base64



model = load_model("pneumonia_model.h5")


# --- Custom background using CSS ---
def set_background(image_file):
    with open(image_file, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()
    css_code = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css_code, unsafe_allow_html=True)

# Call the background function at the beginning of your app
set_background("images/bg.png")  






st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.write("Upload a grayscale chest X-ray image,and the model will predict whether it's PNEUMONIA or NORMAL :)")


uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    
    image_pil = Image.open(uploaded_file).convert("L")  # grayscale
    img_resized = image_pil.resize((150, 150))
    
     
    st.image(img_resized, caption="Uploaded Image",  use_container_width=False, width=300)

    img_array = image.img_to_array(img_resized) 
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0) # (1, 150, 150, 1)
    
    prediction = model.predict(img_array)

    score = prediction[0][0]

    predicted = "NORMAL" if score > 0.5 else "PNEUMONIA"
    
    st.subheader("Prediction Result:")
    if score >0.5 :
       st.success(f"**Predicted: {predicted}**     {(score)*100:.2f}%**")
    else:
        st.success(f"**Predicted: {predicted}**    {(1-score)*100:.2f}%**")

