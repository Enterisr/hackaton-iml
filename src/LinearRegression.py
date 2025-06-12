import numpy as np

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

    def __init__(self):
        super().__init__()
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fit linear model.
        X: 2D numpy array (n_samples, n_features)
        y: 1D numpy array (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
        theta_best = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]

    def predict(self, input_json):
        """
        Predict using the linear model.
        input_json: dict with key 'X' as 2D list or array
        Returns: predictions as numpy array
        """
        X = np.asarray(input_json['X'])
        return X @ self.coef_ + self.intercept_