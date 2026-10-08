import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    X_min, X_max = X[:, 0].min() -1, X[:, 0].max() + 1 # +1 => to add some padding around the data points for better visualization
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(X_min, X_max, 0.005), np.arange(y_min, y_max, 0.005)) #create a grid of points with a step size of 0.005
    grid_points = np.c_[xx.ravel(), yy.ravel()] #flatten the grid points into a 2D array where each row is a point in the grid
    """
        example:
        xx = [[1, 2, 3],
            [1, 2, 3]]
        xx.ravel() => [1, 2, 3, 1, 2, 3]
        yy = [[4, 4, 4],
            [5, 5, 5]]
        yy.ravel() => [4, 4, 4, 5, 5, 5]
        np.c_ => Concatenate along the second axis (columns).
        np.c_[xx.ravel(), yy.ravel()] =>
        [[1, 4],  <-- 1. Pixel (X, Y) 
        [2, 4],   <-- 2. Pixel (X, Y) 
        [3, 4],   <-- 3. Pixel (X, Y) 
        [1, 5],   <-- 4. Pixel (X, Y) 
        [2, 5],   <-- 5. Pixel (X, Y) 
        [3, 5]]   <-- 6. Pixel (X, Y) 
                                        
    """

    Z = model.predict(grid_points) #predict the class labels for each point in the grid
    if len(Z.shape) > 1 and Z.shape[1] > 1:
        Z = np.argmax(Z, axis=1) #if the model outputs probabilities for each class, take the class with the highest probability
    elif len(Z.shape) == 1 or Z.shape[1] == 1:
        Z = (Z > 0.5).astype(int) #if the model outputs a single probability, threshold it at 0.5 to get binary class labels
    Z = Z.reshape(xx.shape) #reshape the predictions to match the shape of the grid
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm) #plot the decision boundary using filled contours
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm') #cmap='coolwarm' => red for class 1, blue for class 0
    
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

if __name__ == "__main__":
    print("Test: Decision Boundary Visualization")
    
    #create a fake model for testing 
    class FakeModel:
        def predict(self, X):
            #A simple fake model that predicts class 1 if the first feature is greater than the second feature, otherwise class 0
            Z = (X[:, 0] > X[:, 1]).astype(int)
            return Z.reshape(-1, 1)

    X_test = np.random.randn(200, 2)    
    #create labels based on a simple rule: class 1 if the first feature is greater than the second feature, otherwise class 0
    y_test = (X_test[:, 0] > X_test[:, 1]).astype(int)
    fake_model = FakeModel()
    plot_decision_boundary(fake_model, X_test, y_test, title="Test: Decision Boundary Visualization")