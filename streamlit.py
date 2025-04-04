# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 20:47:31 2025

@author: Alonso
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline
import kagglehub 
import random
import time
import torch

from openai import OpenAI, RateLimitError   

from tensorflow import keras
from tensorflow.keras.models import load_model

import os

import trial
from PIL import Image as PILImage

# Set the title of your app
st.title('Welcome to the AI corner!🤖')

# Main title for your portfolio
st.title('My python projects 🧑‍💻 ')

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📁 My First Streamlit App", "🖼️ Image Recognition Model", "Chatbot 🤖", "👁️ Recognizer"])

with tab1:

    st.header('Hola!')
    # File uploader for CSV files
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Read the uploaded file into a DataFrame
        df = pd.read_csv(uploaded_file, delimiter= ';')

        # Display the DataFrame
        st.write("Here is your data:")
        st.dataframe(df)

        # Simple operation: Display basic statistics
        if st.checkbox("Show statistics"):
            st.write(df.info())
            st.write(df['Aumenta TB'].describe())
            
            
            x = df['Metabolite']
            y = df['TB promedio'] 
            
            
            fig, ax = plt.subplots(figsize = (15,10))
            ax.bar(x,y)
            ax.set_title('Met vs TB mean')
            ax.set_xlabel('Mets')
            ax.set_ylabel('TB mean')
            st.pyplot(fig)

with tab2:
    st.header('Follow the steps:')
    
    
    # File uploader for image files
    image_file = st.file_uploader("Upload an image of your fingers raised (1 to 5)", type=['jpg', 'jpeg', 'png'])
    
    #model = load_model('ResNet50.h5')
    #model.summary()

    if image_file is not None:
        # Display the uploaded image
        image = Image.open(image_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        
        # Download latest version
        path = kagglehub.model_download("mediapipe/handskeleton/tfjs/default")

        print("Path to model files:", path)
        
        # Load the pre-trained model from Hugging Face
        model = pipeline("image-classification", model="model.json")
        
        # Perform prediction
        predictions = model(image)
        
        # Display the prediction
        st.write("Predicted number of fingers raised:")
        st.write(predictions[0]['label'])
        
with tab3:
    
    st.title(" Chatbot")
    st.write("Welcome to the chatbot! 🤖")
    
    # Read OpenAI API key
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    # Set a default model
    if "openai_model" not in st.session_state:
        print('HOLLAAAAAAA')
        st.session_state["openai_model"] = "gpt-3.5-turbo"
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt}) 
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try :
            
            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                )
                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})

        except:
            #st.error(f"An error occurred: {e}")
            with st.chat_message("assistant"):
                st.markdown("Oops! Sorry, I can't talk now. The max number of tokens has been excedeed. Speak to Alonso.")
        '''except RateLimitError as e:
                if e.code == 'insufficient_quota':
                  st.error("You have exceeded your current quota. Please check your plan and billing details.")'''


with tab4:
    
    st.title("Recognizer 👁️".upper())
    st.write("Welcome! Please, choose a JPG image of an animal or a vegetable.")
    
    image_file = st.file_uploader("Upload a jpg image", type=['jpg', 'jpeg', 'png'])

    if image_file is not None:
        
        try:
            # Display the uploaded image
            image = Image.open(image_file)
            st.write(f"Image format: {image.format}")
            st.write(f"Image size: {image.size}")
            st.write(f"Image mode: {image.mode}")
            st.image(image, caption='Uploaded Image.', use_container_width=True)
            
            # Perform prediction using the function from trial.py
            st.title(f"YOUR IMAGE IS A:  {trial.model.config.id2label[trial.predictor(image)].upper()}")
        
        except ValueError as e:
            st.write(f"Error: your image must be a .jpg , but it is a {image.format}.".upper())
            st.write("Please, upload a jpg image.")