import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import pickle

# Load the CSV file
file_path = r'C:\Users\kn010\OneDrive\Documents\Concordia files\Updated EDA project\combined_data.csv'
data = pd.read_csv(file_path)

# Clean and convert 'Copies sold' to numeric
data['Copies sold'] = data['Copies sold'].replace({' million': '', ',': '', '\xa0': ''}, regex=True)
data['Copies sold'] = pd.to_numeric(data['Copies sold'], errors='coerce')

# Drop rows with NaN values in 'Copies sold'
data = data.dropna(subset=['Copies sold'])

# Feature Engineering
data['Release date'] = pd.to_datetime(data['Release date'], errors='coerce')
data['Release Year'] = data['Release date'].dt.year
data['Release Month'] = data['Release date'].dt.month
data['Sales in Millions'] = data['Copies sold']
franchise_keywords = ['Mario', 'Zelda', 'Final Fantasy', 'Pokemon', 'Call of Duty']
data['Is Franchise'] = data['Game'].apply(lambda x: 1 if any(keyword in x for keyword in franchise_keywords) else 0)
publisher_sales = data.groupby('Publisher')['Copies sold'].transform('sum')
data['Publisher Popularity'] = publisher_sales
data['Log Copies Sold'] = np.log1p(data['Copies sold'])
data = pd.get_dummies(data, columns=['Console_name', 'Genre', 'Publisher'], drop_first=True)

# Drop columns that cannot be used in correlation or modeling
data = data.drop(columns=['Release date', 'Copies sold', 'Game', 'Developer'])

# Impute missing values in the dataset
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Prepare data for predictive modeling
X = data_imputed.drop(columns=['Log Copies Sold', 'Sales in Millions'])
y = data_imputed['Log Copies Sold']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_pred)
print(f'Linear Regression Mean Squared Error: {lr_mse}')

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
print(f'Random Forest Mean Squared Error: {rf_mse}')

# Save the Random Forest model (assuming it's the better model)
model_path = r'C:\Users\kn010\OneDrive\Documents\Concordia files\Updated EDA project\rf_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(rf_model, f)

# Save the imputer
imputer_path = r'C:\Users\kn010\OneDrive\Documents\Concordia files\Updated EDA project\imputer.pkl'
with open(imputer_path, 'wb') as f:
    pickle.dump(imputer, f)
