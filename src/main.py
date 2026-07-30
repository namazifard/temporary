import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import add from utils

# Load the dataset
data = pd.read_csv('data.csv')

def function1(data):
    # Perform some data preprocessing
    data = data.dropna()  # Drop missing values
    return data

print(add(5, 3))  # Example usage of the add function