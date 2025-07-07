import streamlit as st
import random
import joblib
import pickle

@st.cache_data
def generate_random_value(x): 
    return random.uniform(0, x) 

a = generate_random_value(10) 
b = generate_random_value(20) 

st.write(a) 
st.write(b)

clf_jbl = joblib.load("model.jbl")
clf_pkl = pickle.load(open("model.pkl", 'rb'))