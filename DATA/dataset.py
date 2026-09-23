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

def split_dataset(X, y, train_ratio, validation_ratio, seed=None):
    test_ratio = (1 - train_ratio - validation_ratio)

    sample_count = len(X)
    train_count = int(train_ratio * sample_count)
    validation_count = int(validation_ratio * sample_count)
    test_count = sample_count - train_count - validation_count

    rng = np.random.default_rng(seed) # Create a random number generator with the provided seed
    indices = rng.permutation(sample_count) 
    train_indices = indices[:train_count]
    validation_indices = indices[train_count:validation_count + train_count]
    test_indices = indices[train_count + validation_count:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_validation = X[validation_indices]
    y_validation = y[validation_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    #CONTROL#
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")
    if not 0 <= test_ratio < 1:
        raise ValueError("test_ratio must be less than 1.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if train_ratio > 0 and train_count == 0:
        raise ValueError("Training set would be empty.")
    if validation_ratio > 0 and validation_count == 0:
        raise ValueError("Validation set would be empty.")
    if test_count == 0:
        raise ValueError("Test set would be empty.")


    return (X_train, y_train, X_validation, y_validation, X_test, y_test)

if __name__ == "__main__":
    ##Test##
    X = np.arange(300).reshape(100, 3) # Create a 100x3 array of integers from 0 to 299
    y = np.arange(100).reshape(100, 1)

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X,
        y,
        train_ratio=0.70,
        validation_ratio=0.15,
        seed=42
    )
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print(
        "Total samples:",
        len(X_train) + len(X_val) + len(X_test)
    )
    print(np.array_equal(X_train[:, 0] // 3, y_train[:, 0]))