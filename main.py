import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('data.csv')

def function1(data):
    # Perform some data preprocessing
    data = data.dropna()  # Drop missing values
    return data

