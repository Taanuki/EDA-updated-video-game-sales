import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'combined_data.csv'
data = pd.read_csv(file_path)

# Data Preprocessing
data['Copies sold'] = data['Copies sold'].replace({' million': '', ',': '', '\xa0': ''}, regex=True)
data['Copies sold'] = pd.to_numeric(data['Copies sold'], errors='coerce')
data['Release date'] = pd.to_datetime(data['Release date'], errors='coerce
