# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 20:47:31 2025

@author: Alonso
"""

import streamlit as st
import pandas as pd

# Set the title of your app
st.title('My First Streamlit App')

# File uploader for CSV files
uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

# Check if a file is uploaded
if uploaded_file is not None:
    # Read the uploaded file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the DataFrame
    st.write("Here is your data:")
    st.dataframe(df)

    # Simple operation: Display basic statistics
    if st.checkbox("Show statistics"):
        st.write(df.describe())
