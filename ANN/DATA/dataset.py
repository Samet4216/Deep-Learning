import numpy as np


def create_dataset(): #return x and y for creating a dataset
    X = np.array([
        [70, 24, 10],
        [75, 24, 12],
        [80, 23, 15],
        [85, 23, 17],
        [90, 22, 20]
        ])

    y = np.array([
        [1500],
        [1800],
        [2200],
        [2600],
        [3000]
        ])
    return X, y
if __name__ == "__main__":
    X, y = create_dataset()
    print("Input data (X):")
    print(X)
    print("Output data (y):")
    print(y)