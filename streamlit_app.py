# -*- coding: utf-8 -*-
"""
Created on 30/05/2024

@author: 
         *   Abdallah Raed Hani Ghordlo
         *   Hussein Abdikarim Hussein
         *   Fevzi Berk Çeliktaş
         *   Melih Aydın
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
#sklearn
import sklearn
import imblearn

# Print library versions
st.write(f"Streamlit version: {st._version_}")
st.write(f"Pandas version: {pd._version_}")
st.write(f"Numpy version: {np._version_}")
st.write(f"Scikit-learn version: {sklearn._version_}")
st.write(f"Imbalanced-learn version: {imblearn._version_}")

# loading the saved model
loaded_model = pickle.load(open('bank_model.pkl.sav', 'rb'))

# Function for prediction
def prediction_function(input_data):
    # Convert input data to numpy array and reshape
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

def main():
    # Title of the Streamlit app
    st.title('Bank Marketing Prediction Web App')

    # Collecting additional input data from the user
    age = st.number_input('Age', min_value=18, max_value=100, step=1)
    job = st.selectbox('Job', ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", 
                               "retired", "self-employed", "services", "student", "technician", 
                               "unemployed", "unknown"])
    marital = st.selectbox('Marital Status', ["divorced", "married", "single", "unknown"])
    education = st.selectbox('Education', ["basic.4y", "basic.6y", "basic.9y", "high.school", 
                                           "illiterate", "professional.course", "university.degree", 
                                           "unknown"])
    default = st.selectbox('Default', ["no", "yes", "unknown"])
    housing = st.selectbox('Housing Loan', ["no", "yes", "unknown"])
    loan = st.selectbox('Personal Loan', ["no", "yes", "unknown"])
    contact = st.selectbox('Contact Communication Type', ["cellular", "telephone", "unknown"])
    month = st.selectbox('Month of Contact', ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", 
                                              "sep", "oct", "nov", "dec"])
    day_of_week = st.selectbox('Day of Week', ["mon", "tue", "wed", "thu", "fri"])
    campaign = st.number_input('Campaign', min_value=1)
    pdays = st.number_input('Pdays', min_value=-1)
    poutcome = st.selectbox('Poutcome', ["failure", "nonexistent", "success"])
    cons_price_idx = st.number_input('Consumer Price Index', value=0.0)
    cons_conf_idx = st.number_input('Consumer Confidence Index', value=0.0)

   # These columns have an affect on our model
    duration = st.number_input('Duration', min_value=0)
    previous = st.number_input('Previous', min_value=0)
    emp_var_rate = st.number_input('Employment Variation Rate', value=0.0)
    euribor3m = st.number_input('Euribor 3 Month Rate', value=0.0)
    nr_employed = st.number_input('Number of Employed', value=0.0)
    contacted_before = st.number_input('Contacted Before (0 or 1)', min_value=0, max_value=1, step=1)

    # Code for prediction
    diagnosis = ''
    
    # Button for prediction
    if st.button('Predict'):
        input_data = [duration, previous, emp_var_rate, euribor3m, nr_employed, contacted_before]
        diagnosis = prediction_function(input_data)
    
    st.success(f'The prediction is: {diagnosis}')

if __name__ == '__main__':
    main()
