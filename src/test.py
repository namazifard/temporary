import pandas as pd
import numpy as np

def test_dataframe_creation():
    # Create a sample DataFrame
    data = {
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    }
    df = pd.DataFrame(data)

    # Check if the DataFrame is created correctly
    assert df.shape == (3, 3), "DataFrame shape is incorrect"
    assert list(df.columns) == ['A', 'B', 'C'], "DataFrame columns are incorrect"
    assert df['A'].tolist() == [1, 2, 3], "Column A values are incorrect"
    assert df['B'].tolist() == [4, 5, 6], "Column B values are incorrect"
    assert df['C'].tolist() == [7, 8, 9], "Column C values are incorrect"   

def test_dataframe_operations():
    # Create a sample DataFrame
    data = {
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    }
    df = pd.DataFrame(data)

    # Test addition operation
    df['D'] = df['A'] + df['B']
    assert df['D'].tolist() == [5, 7, 9], "Addition operation is incorrect"

    # Test multiplication operation
    df['E'] = df['A'] * df['C']
    assert df['E'].tolist() == [7, 16, 27], "Multiplication operation is incorrect"

    # Test filtering operation
    filtered_df = df[df['A'] > 1]
    assert filtered_df.shape == (2, 5), "Filtering operation is incorrect"
    assert filtered_df['A'].tolist() == [2, 3], "Filtered values are incorrect" 