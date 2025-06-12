import numpy as np
import os
import json
class BaseModel:
    """
    A base class for all models
    """
    counter = 0  # a static variable that counts number of models

    def __init__(self):
        BaseModel.counter += 1
        self.name = self.__class__.__name__ + f"_{BaseModel.counter}"

    def predict(self, input_json):
        """
        Implement this function in inheriting classes
        """
        raise NotImplementedError("Subclasses must implement this method.")


class LinearRegression(BaseModel):
    """
    Simple Linear Regression implementation
    """

    def __init__(self, weights_path=None):
        super().__init__()
        self.coef_ = None
        self.intercept_ = None
        
        # Load weights if path is provided
        if weights_path:
            try:
                self.load_weights(weights_path)
            except Exception as e:
                print(f"Warning: Failed to load weights from {weights_path}: {e}")

    def fit_or_load(self, X, y, weights_path=None, force_train=False):
        """
        Fit linear model.
        X: 2D numpy array (n_samples, n_features)
        y: 1D numpy array (n_samples,)
        weights_path: Optional path to pre-trained weights
        force_train: If True, train the model even if weights were loaded
        """
        # Try to load weights first if path is provided
        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)
            if not force_train:
                print(f"Loaded pre-trained weights from {weights_path}, skipping training")
                return self
    
        # Train the model
        X = np.asarray(X)
        y = np.asarray(y)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
        theta_best = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]
        
        # Save weights if path is provided
        if weights_path:
            self.save_weights(weights_path)
            
        return self

    def predict(self, input_json):
        """
        Predict using the linear model.
        input_json: dict with key 'X' as 2D list or array
        Returns: predictions as numpy array
        """
        X = np.asarray(input_json['X'])
        return X @ self.coef_ + self.intercept_
        
    def save_weights(self, filepath):
        """
        Save model weights to a file
        
        Parameters:
        -----------
        filepath : str
            Path where to save the weights
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save weights in JSON format
        weights = {
            'intercept': float(self.intercept_),
            'coefficients': self.coef_.tolist() if hasattr(self.coef_, 'tolist') else list(self.coef_)
        }
        
        with open(filepath, 'w') as f:
            json.dump(weights, f, indent=2)
        
        return filepath
        
    def load_weights(self, filepath):
        """
        Load model weights from a file
        
        Parameters:
        -----------
        filepath : str
            Path to the weights file
        """
        with open(filepath, 'r') as f:
            weights = json.load(f)
            
        self.intercept_ = weights['intercept']
        self.coef_ = np.array(weights['coefficients'])
        
        return self