import streamlit as st
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.pipeline.prediction_pipeline import PredictionPipeline


st.title ("Student Performance Predictor")

target_variable = st.selectbox(
        "Which one you want to predict",
        ('math_score', 'writing_score', 'reading_score'),
        placeholder="Select feature...",
    )
all_features = ['math_score', 'reading_score', 'writing_score']
input_features = [f for f in all_features if f != target_variable]
col_1, col_2= st.columns(2)

    
  

with col_1:
    gender = st.selectbox(
        "Gender",
        ("female","male"),
        placeholder="select gender",
    )
    race_ethnicity = st.selectbox(
        "Race or Ethnicity",
        ('group B', 'group C', 'group A' ,'group D', 'group E'),
        placeholder= "select race_ethnicity type",
    )

    parental_level_of_education = st.selectbox(
        "Parent's Enducation",
        ("bachelor's degree", 'some college' ,"master's degree", "associate's degree",'high school' ,'some high school'),
        placeholder= "select Parent's education",
    )

    lunch = st.selectbox(
        "Lunch",
        ('standard' ,'free/reduced'),
        placeholder = "Select Lunch type",
    )
    test_preparation_course = st.selectbox(
        "Test Preparation Course",
        ('none' ,'completed'),
        placeholder = "Select Test preparation Course"
    )
    input_values = {}
    for feature in input_features:
        input_values[feature] = st.number_input(f"Enter {feature.replace('_', ' ').capitalize()}", min_value=0, max_value=100)
    
    logging.info("all required fields have value")

with col_2:
    if st.button("Predict"):
        try:
            logging.info("predicting the result")

            features = pd.DataFrame([{
            'gender': gender,
            'race_ethnicity': race_ethnicity,
            'parental_level_of_education': parental_level_of_education,
            'lunch': lunch,
            'test_preparation_course': test_preparation_course,
            **input_values
        }])
            
            prediction_pipeline = PredictionPipeline()
            predict = prediction_pipeline.predict(features,target_variable=target_variable)

            st.success(f"Predicted {target_variable.replace('_', ' ').capitalize()}: **{round(predict[0], 2)}**")
        except Exception as e:
            raise CustomException(e, sys)