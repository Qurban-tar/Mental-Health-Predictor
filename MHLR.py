"""This project focuses on predicting a person's mental health score using a simple linear regression model. 
The goal is to estimate how healthy or stressed someone might feel, based on their daily digital habits.
Specifically, the model takes three main input features: the number of hours a person spends on screens per day,
the average number of hours they sleep, and the time they spend on social media. These features are used to predict
a mental health score on a scale from 1 to 10, where 1 indicates very poor mental health and 10 indicates excellent mental health.
To build this project, Python programming language was used along with essential data science libraries 
such as Pandas for data manipulation, scikit-learn for building and training the machine learning model, 
and matplotlib and seaborn for visualizing the data. The process involved loading and exploring the dataset,
training a linear regression model on the data, and then evaluating the model’s performance using metrics like 
the R² score and mean squared error. A visualization comparing actual and predicted values was also created to 
better understand how well the model performs. This project is a relevant and practical demonstration of how 
simple data about our digital life can be used to assess mental well-being."""



import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Page Title
st.title("Mental Health Score Predictor")

st.write("""
Enter your daily habits below to get your predicted **Mental Health Score (1 to 10)**.
""")

# Load dataset
df = pd.read_csv("mental_health_data.csv")

# Prepare features and target
X = df[['screen_time_hours', 'sleep_hours', 'social_media_hours']]
y = df['mental_health_score']

# Train model
model = LinearRegression()
model.fit(X, y)

# Input fields (user types values directly)
screen_time = st.text_input("Screen Time (hours per day)", value="6.0")
sleep_hours = st.text_input("Sleep Hours (hours per day)", value="7.0")
social_media = st.text_input("Social Media Hours (per day)", value="3.0")

# Predict button
if st.button("Predict Mental Health Score"):
    try:
        # Convert input strings to floats
        input_data = np.array([[float(screen_time), float(sleep_hours), float(social_media)]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        st.subheader("Predicted Mental Health Score:")
        st.success(f"{prediction:.2f} / 10")
    except ValueError:
        st.error("Please enter valid numeric values in all fields.")
