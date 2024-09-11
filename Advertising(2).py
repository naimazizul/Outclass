import streamlit as st
import pandas as pd
import seaborn as sns

st.write("# Advertising")
st.write("This app predicts the **Sales**!")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 0,300,147)
    Radio = st.sidebar.slider(Radio, 0,50,25)
    Newspaper = st.sidebar.slider('Newspaper', 0,115,30)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper,}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("Advertising(2).h5", "rb")) #rb: read binary
new_pred = loaded_model.predict(df) # testing (examination)

st.subheader('Prediction')
st.write(new_pred)
