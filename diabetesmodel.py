# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import streamlit as st
import os

# Create the full path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'trainedmodel.sav')

# Load the model
try:
    model = pickle.load(open(model_path, 'rb'))
    st.write("Model loaded successfully.")
except FileNotFoundError:
    st.write("Model file not found. Please check the file path.")
except Exception as e:
    st.write(f"An error occurred: {e}")

input_data=(4,110,92,0,0,37.6,0.191,30)
input_numpy=np.asarray(input_data)
input_reshaped=input_numpy.reshape(1,-1)

prediction=model.predict(input_reshaped)
print(prediction)
if(prediction[0]==1):
  print("Diabetic")
else:
  print("Not Diabetic")  