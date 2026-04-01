import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import model_engine

def main():
    st.set_page_config(
        page_title="Image Recognizer | CIFAR-10",
        layout="wide"
    )

    st.sidebar.title("Model Details")
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/2/21/Tensorflow_logo.svg", width=50)
    st.sidebar.markdown("""
    **Model:** Convolutional Neural Network (CNN)  
    **Dataset:** CIFAR-10  
    **Categories:** 10  
    **Input Size:** 32x32 pixels  
    
    ---
    ### Classes it can recognize:
    - Airplane
    - Automobile
    - Bird
    - Cat
    - Deer
    - Dog
    - Frog
    - Horse
    - Ship
    - Truck
    """)
    
    st.sidebar.divider()
    st.sidebar.caption("v2.0")

    st.title("Image Recognition")
    st.write("Upload an image for analysis.")

    try:
        model = model_engine.load_cifar10_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            
            with st.spinner("Analyzing image..."):
                img_array = model_engine.preprocess_image(image)
                predictions = model_engine.predict_image(model, img_array)
            
            indices = predictions.argsort()[-3:][::-1]
            
            top_classes = [model_engine.CIFAR10_CLASSES[i] for i in indices]
            top_probs = [predictions[i] * 100 for i in indices]
            
            max_prob = top_probs[0]
            predicted_label = top_classes[0]

            df_results = pd.DataFrame({
                'Class': top_classes,
                'Confidence (%)': top_probs
            })

            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                st.subheader("Uploaded Image")
                st.image(image, use_container_width=True, caption=f"Format: {image.format}")

            with col2:
                st.subheader("Prediction Analysis")
                
                if (max_prob / 100) < 0.65:
                    st.warning("Low Confidence Result")
                    st.write("The model is not highly confident in this prediction.")
                else:
                    st.success(f"Top Match: {predicted_label}")

                fig = px.bar(
                    df_results,
                    x='Confidence (%)',
                    y='Class',
                    orientation='h',
                    text='Confidence (%)',
                    color='Confidence (%)',
                    color_continuous_scale='Greens',
                    labels={'Confidence (%)': 'Probability %'}
                )
                
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig.update_layout(
                    showlegend=False,
                    yaxis={'categoryorder': 'total ascending'},
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

    else:
        st.info("Please upload an image to start.")
        
        st.write("---")
        st.caption("Available categories:")
        cols = st.columns(5)
        for i, cat in enumerate(model_engine.CIFAR10_CLASSES[:5]):
            cols[i].button(cat, disabled=True)

if __name__ == "__main__":
    main()
