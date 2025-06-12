import numpy as np
import pandas as pd
from src.dynamic_draft import DynamicLinearRegression
from src.preprocess import INTRESTING_FEATURES

def test_dynamic_regression():
    # Create a model
    print("Creating model...")
    model = DynamicLinearRegression()
    
    # Create sample data
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    
    # Test fit method
    print("Testing fit method...")
    model.fit(X, y)
    
    # Create a test input with sample player data
    player_data = {}
    for feature in INTRESTING_FEATURES:
        player_data[feature] = [10, 20]
    
    player_data['personId'] = [123, 456]
    
    test_input = {
        'draft class': [123, 456],
        'last season': pd.DataFrame(player_data)
    }
    
    # Test predict method
    print("Testing predict method...")
    try:
        prediction = model.predict(test_input)
        print(f'Prediction: {prediction}')
        print("Test passed!")
    except Exception as e:
        print(f'Error in predict method: {str(e)}')

if __name__ == "__main__":
    test_dynamic_regression()
