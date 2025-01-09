import streamlit as st
import requests


def main():
    API_URL = "http://127.0.0.1:8000/classify"

    st.title("Image Classification")
    st.write("Upload an image, and this app will classify it using ResNet50.")
    st.sidebar.header("Predictions:")
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Send the image to the API
        with st.sidebar:
            with st.spinner("Classifying..."):
                try:
                    with st.spinner("Processing image..."):
                        image_bytes = uploaded_file.read()
                        files = {"image_file": (uploaded_file.name, image_bytes, uploaded_file.type)}
                        response = requests.post(API_URL, files=files)
                        response.raise_for_status()
                        predictions = response.json()["predictions"]
                        # Format and Show predictions
                        for pred in predictions:
                            st.sidebar.write(f"**{pred['class']}**: {pred['probability']:.2f}")
                except Exception as e:
                    st.sidebar.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()