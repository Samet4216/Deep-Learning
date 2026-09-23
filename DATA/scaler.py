import numpy as np


class MinMaxScaler:
    def __init__(self):
        self.epsilon = 1e-8
        self.min = None
        self.max = None
        self.range = None

    def fit(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        self.range = self.max - self.min
        return self # fit() is not to output the numbers, but to store them inside the object.include min and max inside the object.
        # print(scaler.min, scaler.max) will give you the min and max values.

    # note: before transforming, we need to fit the scaler first. fit() and after that, transform()
    def transform(self, X):
        if self.max is None:
            raise ValueError("Scaler not fitted. Please fit the scaler first.")
        return (X-self.min) / (self.range + self.epsilon)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, scaled_X):
        if self.min is None:
            raise ValueError("Scaler not fitted. Please fit the scaler first.")
        return scaled_X * (self.range + self.epsilon) + self.min


class StandardScaler:
    def __init__(self):
        self.epsilon = 1e-8
        self.mean = None
        self.std = None
        
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) # std is the standard deviation of the data.
        return self 

    # note: before transforming, we need to fit the scaler first. fit() and after that, transform()
    def transform(self, X):
        if self.std is None:
            raise ValueError("Scaler not fitted. Please fit the scaler first.")
        return (X-self.mean) / (self.std + self.epsilon)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, scaled_X): # inverse_transform() is to transform the scaled data back to the original data.
    #ex. [1000,5000]=>[0,1] and 3000=>0.5 we need to transform 0.5 back to 3000.
        if self.std is None:
            raise ValueError("Scaler not fitted. Please fit the scaler first.")
        return scaled_X * (self.std + self.epsilon) + self.mean