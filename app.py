import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("✍️ Handwritten Digit Recognizer")

model = tf.keras.models.load_model("model.h5")

file = st.file_uploader("Upload a 28x28 digit image", type=["png", "jpg"])

if file:
    img = Image.open(file).convert("L").resize((28, 28))
    st.image(img, caption="Uploaded Image", width=150)

    x = np.array(img) / 255.0
    x = x.reshape(1, 28, 28, 1)
    pred = model.predict(x)
    st.success(f"Predicted Digit: {np.argmax(pred)}")
