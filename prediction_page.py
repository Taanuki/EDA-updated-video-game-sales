import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and imputer
model_path = r'C:\Users\kn010\OneDrive\Documents\Concordia files\Updated EDA project\rf_model.pkl'
imputer_path = r'C:\Users\kn010\OneDrive\Documents\Concordia files\Updated EDA project\imputer.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(imputer_path, 'rb') as f:
    imputer = pickle.load(f)

def show_prediction_page():
    st.title("Video Game Sales Prediction")

    release_year = st.number_input("Release Year", min_value=1980, max_value=2024, value=2024)
    release_month = st.number_input("Release Month", min_value=1, max_value=12, value=1)
    is_franchise = st.selectbox("Is Franchise?", [0, 1])
    publisher_popularity = st.number_input("Publisher Popularity", min_value=0.0, value=50.0)

    console_names = ['Console_name_PS2', 'Console_name_PS3', 'Console_name_PS4', 'Console_name_Xbox', 'Console_name_Xbox360', 'Console_name_XboxOne', 'Console_name_Wii', 'Console_name_WiiU', 'Console_name_Switch', 'Console_name_PC', 'Console_name_NDS', 'Console_name_3DS', 'Console_name_Vita']
    genres = ['Genre_Action', 'Genre_Adventure', 'Genre_Fighting', 'Genre_Misc', 'Genre_Platform', 'Genre_Puzzle', 'Genre_Racing', 'Genre_Role-Playing', 'Genre_Shooter', 'Genre_Simulation', 'Genre_Sports', 'Genre_Strategy']
    publishers = ['Publisher_Electronic Arts', 'Publisher_Nintendo', 'Publisher_Sony Computer Entertainment', 'Publisher_Activision', 'Publisher_Take-Two Interactive', 'Publisher_Capcom', 'Publisher_Konami Digital Entertainment', 'Publisher_Sega', 'Publisher_Square Enix', 'Publisher_Ubisoft']

    console_values = {name: st.selectbox(name, [0, 1]) for name in console_names}
    genre_values = {name: st.selectbox(name, [0, 1]) for name in genres}
    publisher_values = {name: st.selectbox(name, [0, 1]) for name in publishers}

    input_features = {
        'Release Year': release_year,
        'Release Month': release_month,
        'Is Franchise': is_franchise,
        'Publisher Popularity': publisher_popularity,
        **console_values,
        **genre_values,
        **publisher_values
    }

    input_df = pd.DataFrame([input_features])

    input_df_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)

    if st.button("Predict"):
        prediction = model.predict(input_df_imputed)
        st.write(f"### Predicted Log Copies Sold: {prediction[0]}")
        st.write(f"### Predicted Copies Sold: {np.expm1(prediction[0])}")
