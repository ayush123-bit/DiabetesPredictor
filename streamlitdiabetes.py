# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:37:30 2024

@author: ayush
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


# Define the prediction function
def diabetes_prediction(input_data):
    # Convert input_data to a numpy array
    input_numpy = np.asarray(input_data)
    # Reshape the array for prediction (1 instance, multiple features)
    input_reshaped = input_numpy.reshape(1, -1)
    
    # Make the prediction
    prediction = model.predict(input_reshaped)
    
    if prediction[0] == 1:
        return "Diabetic"
    else:
        return "Not Diabetic"

# Main function for the Streamlit web app
def main():
    # Set the title of the web app
    st.title("Diabetes Prediction Web App")
    
    # Get user inputs
    pregnancies = st.text_input("Number of pregnancies", "0")
    glucose = st.text_input("Glucose Level", "0")
    bp = st.text_input("Blood Pressure", "0")
    skinthick = st.text_input("Skin Thickness", "0")
    insulin = st.text_input("Insulin", "0")
    bmi = st.text_input("BMI", "0")
    diabetes_pedigree_function = st.text_input("Diabetes Pedigree Function", "0")
    age = st.text_input("Age", "0")
    
    # Initialize the diagnosis variable
    diagnosis = ""
    
    # Button for prediction
    if st.button("Result"):
        # Convert inputs to proper types
        input_data = [int(pregnancies), float(glucose), float(bp), float(skinthick), float(insulin), 
                      float(bmi), float(diabetes_pedigree_function), int(age)]
        
        # Call the prediction function
        diagnosis = diabetes_prediction(input_data)
        
        # Show the result
        st.success(diagnosis)

# Run the main function when the script is executed
if __name__ == '__main__':
    main()
