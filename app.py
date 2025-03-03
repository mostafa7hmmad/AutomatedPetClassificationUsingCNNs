
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = 'my_cnn_model.h5'
CLASS_INDICES_PATH = 'class_indices.npy'

def verify_files():
    """Check if required files exist"""
    missing_files = []
    if not os.path.exists(MODEL_PATH):
        missing_files.append(MODEL_PATH)
    if not os.path.exists(CLASS_INDICES_PATH):
        missing_files.append(CLASS_INDICES_PATH)
    
    if missing_files:
        st.error(f"Missing required files: {', '.join(missing_files)}")
        st.write("Please ensure you have:")
        st.write("- A trained model saved as 'my_model.h5'")
        st.write("- Class indices saved as 'class_indices.npy'")
        return False
    return True

@st.cache_resource
def load_model_and_indices():
    """Load model and class indices with verification"""
    if not verify_files():
        st.stop()
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        class_indices = np.load(CLASS_INDICES_PATH, allow_pickle=True).item()
        return model, class_indices
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def preprocess_image(image_file):
    """Process uploaded image for model prediction"""
    try:
        img = Image.open(image_file).convert('RGB')
        img = img.resize((64, 64))  # Match model input size
        img_array = np.array(img) / 255.0  # Normalize
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.stop()

def main():
    st.title("🐱 Cat vs Dog Classifier 🐶")
    st.markdown("""
    ### Instructions:
    1. Upload an image of a cat or dog (JPEG/PNG)
    2. Wait for prediction
    3. View results!
    """)

    # Load assets early to verify existence
    model, class_indices = load_model_and_indices()
    class_names = list(class_indices.keys())

    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )

    if uploaded_file:
        try:
            # Display image in a collapsible section
            with st.expander("Uploaded Image", expanded=True):
                image = Image.open(uploaded_file)
                st.image(image, 
                        caption="Uploaded Image", 
                        use_container_width=True)

            # Process and predict
            with st.spinner('Analyzing image...'):
                processed_image = preprocess_image(uploaded_file)
                prediction = model.predict(processed_image)[0][0]

            # Display results
            confidence = prediction if prediction > 0.5 else 1 - prediction
            predicted_class = class_names[1] if prediction > 0.5 else class_names[0]

            st.subheader("Results:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", predicted_class)
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            
            st.progress(float(confidence))

            # Show debug info in another expander
            with st.expander("Technical Details", expanded=False):
                st.write(f"Raw prediction value: {prediction:.4f}")
                st.write(f"Class mapping: {class_indices}")
                st.write(f"Model input shape: {model.input_shape}")
                st.write(f"Model summary:")
                st.text(model.summary())

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

if __name__ == "__main__":
    main()