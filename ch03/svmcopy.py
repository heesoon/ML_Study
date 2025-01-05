import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LinearSVM:
    def __init__(self, eta=0.001, n_iter=1000, C=1):
        self.eta = eta  # Learning rate
        self.n_iter = n_iter  # Number of iterations
        self.C = C  # Regularization parameter

    def fit(self, X, y):
        # Number of samples and features
        m, n = X.shape
        
        # Initialize weights and bias to zeros
        self.w = np.zeros(n)
        self.b = 0

        # Gradient descent loop
        for i in range(self.n_iter):
            # Initialize gradients
            dw = np.zeros(n)
            db = 0

            # Loop over each sample in the dataset
            for j in range(m):
                # Compute the margin for the sample
                margin = y[j] * (np.dot(X[j], self.w) + self.b)
                
                # If the margin is less than 1, we need to update the weights
                if margin < 1:
                    dw -= self.C * y[j] * X[j]  # Regularization + hinge loss gradient
                    db -= self.C * y[j]  # Bias gradient

            # Update weights and bias
            self.w -= self.eta * dw / m
            self.b -= self.eta * db / m

        return self

    def predict(self, X):
        # Return predictions based on the sign of the decision function
        return np.sign(np.dot(X, self.w) + self.b)

    def decision_function(self, X):
        # Return decision function (for margin)
        return np.dot(X, self.w) + self.b

# Example: Using synthetic data for testing
def main():
    # Create a simple binary classification dataset
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, random_state=42)
    
    # Convert labels to -1, 1 (SVM convention)
    y = 2 * y - 1

    # Standardize the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the SVM model using gradient descent
    svm = LinearSVM(eta=0.001, n_iter=1000, C=1)
    svm.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Plot decision boundary
    plot_decision_boundary(X, y, svm)

# Function to plot the decision boundary
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50, cmap=plt.cm.Paired)
    plt.show()

if __name__ == '__main__':
    main()
