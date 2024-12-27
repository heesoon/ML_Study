import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class SVM(object):
    def __init__(self, eta=0.05, n_iter=100, C=1):
        self.eta = eta
        self.n_iter = n_iter
        self.w = []
        self.b = []
        self.C = C

    def init_weight(self, X):
        self.w = np.zeros(X.shape[1]) # number of characteristic

    def update_weight(self, dw, db):
        self.w[1:] -= self.eta * dw
        self.w[0] -= self.eta * db

    def hinge_loss(self, X, y):
        print('X : ', X)
        print(self.w[1:])
        return y * (np.dot(X, self.w[1:]) + self.w[0])
    
    def activation(self, z):
        return z

    def fit(self, X, y):
        self.init_weight(X)
        for i in range(self.n_iter):
            for idx, xi in enumerate(X):
                if self.hinge_loss(xi, y[idx]) < 1:
                    dw = self.w - self.C * y[idx]*xi
                    db = -self.C * y[idx]
                else:
                    dw = self.w
                    db = 0
                self.update_weight(dw, db)
        print('optimum w :', self.w)
        print('optimum b :', self.b)
        return self
    
#    def predict(self, X):
#        return np.where(self.net_input(X) >= 0.0, 1, 0)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors=colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolors='black')
    
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:,  0], X_test[:, 1],
                    facecolors='none', edgecolors='black', alpha=1.0,
                    linewidths=1, marker='o',
                    s=100, label='test_set')
    
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    sc = StandardScaler()
    sc.fit(X_train)
    
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
    
    svm = SVM()
    svm.fit(X_train_01_subset, y_train_01_subset)
    #plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=lrgd)

if __name__ == '__main__':
    main()
