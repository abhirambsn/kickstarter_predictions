from tensorflow import keras
import streamlit as st
import joblib
import numpy as np
import pandas as pd

@st.cache_resource()
def load_model(model_path):
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_result(fail_val, success_val):
    if fail_val < success_val: return "Can be Successful"
    return "No Guarantee of being Successful"

# UI
st.title("Kickstarter Campaign success Predictor")

with st.spinner("Loading Model..."):
    model = load_model("./kickstarter_predictions.h5")

st.write("""
## Input your parameters and get the predicted result
""")

backers = st.number_input("Backers", step=1, value=0, min_value=0)
launch_date = st.date_input("Campaign launch date")
deadline = st.date_input("Campaign deadline")

goal = st.number_input("Goal")
pledged = st.number_input("Amount Pledged")


def calculate_params(backers, launch_date, deadline, goal, pledged):
    scaler = joblib.load("scaler.gz")
    print(scaler.get_params())
    pct = pledged / goal
    diff = deadline - launch_date
    data = pd.DataFrame([[backers, diff.days, pct]], columns=['Backers', 'Days', 'Percentage'])
    data[['Days', 'Backers']] = scaler.transform(data[['Days', 'Backers']])
    with st.spinner("Predicting..."):
        pred = model.predict(data)
        print(pred)
        st.write("Prediction Result: " + get_result(pred[0][0], pred[0][1]))
    

st.button("Predict Result", on_click=calculate_params, args=(backers, launch_date, deadline, goal,pledged))    