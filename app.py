from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and imputer
model_path = r'C:\Users\kn010\OneDrive\Documents\Concordia files\Updated EDA project\rf_model.pkl'
imputer_path = r'C:\Users\kn010\OneDrive\Documents\Concordia files\Updated EDA project\imputer.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(imputer_path, 'rb') as f:
    imputer = pickle.load(f)

# Explicitly define the columns for Console_name, Genre, and Publisher
console_columns = [
    'Console_name_PS2', 'Console_name_PS3', 'Console_name_PS4', 'Console_name_Xbox', 
    'Console_name_Xbox360', 'Console_name_XboxOne', 'Console_name_Wii', 'Console_name_WiiU', 
    'Console_name_Switch', 'Console_name_PC', 'Console_name_NDS', 'Console_name_3DS', 
    'Console_name_Vita'
]

genre_columns = [
    'Genre_Action', 'Genre_Adventure', 'Genre_Fighting', 'Genre_Misc', 'Genre_Platform', 
    'Genre_Puzzle', 'Genre_Racing', 'Genre_Role-Playing', 'Genre_Shooter', 'Genre_Simulation', 
    'Genre_Sports', 'Genre_Strategy'
]

publisher_columns = [
    'Publisher_Electronic Arts', 'Publisher_Nintendo', 'Publisher_Sony Computer Entertainment', 
    'Publisher_Activision', 'Publisher_Take-Two Interactive', 'Publisher_Capcom', 
    'Publisher_Konami Digital Entertainment', 'Publisher_Sega', 'Publisher_Square Enix', 
    'Publisher_Ubisoft'
]

# Define a route for the default URL, which loads the form
@app.route('/')
def form():
    return """
        <form action="/predict" method="post">
            <label for="features">Input Features:</label><br><br>
            <textarea id="features" name="features" rows="10" cols="50"></textarea><br><br>
            <input type="submit" value="Predict">
        </form>
    """

# Define a route for the API that will process the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Parse input features
    input_features = request.form['features']
    input_data = [list(map(float, feature.split(','))) for feature in input_features.split('\n') if feature]

    # Convert input data to DataFrame
    columns = ['Release Year', 'Release Month', 'Is Franchise', 'Publisher Popularity'] + console_columns + genre_columns + publisher_columns

    input_df = pd.DataFrame(input_data, columns=columns)

    # Impute missing values
    input_df_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)

    # Predict using the trained model
    predictions = model.predict(input_df_imputed)
    
    # Return the predictions as JSON
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
