from DATA.dataset import create_dataset 

X, y = create_dataset()


print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Number of targets:", y.shape[1])