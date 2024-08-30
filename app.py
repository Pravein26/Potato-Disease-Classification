import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('model.h5')

# Define the labels
labels = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Set the title of the web app
st.title("Potato Disease Classification")

# Allow users to upload images
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Convert the uploaded image to a PIL Image
    img = Image.open(uploaded_file)

    # Display the image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Resize the image to the target size
    img = img.resize((224, 224))

    # Convert the image to a numpy array and normalize it
    img_array = np.array(img) / 255.0

    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)

    # Get the index of the highest prediction
    max_index = np.argmax(predictions)

    # Display the prediction
    st.write(f"Prediction: {labels[max_index]}")
