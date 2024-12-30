import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from collections import Counter

class KNN(object):
    def __init__(self, k=3):
        self.k = k
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
        #return np.sqrt(np.sum(np.abs(x1 - x2)))

    def fit(self, Xtrain, ytrain):
        self.Xtrain = Xtrain
        self.ytrain = ytrain

    def _predict(self, x):
        distances = [self.euclidean_distance(x, xtrain) for xtrain in self.Xtrain]
        kIndices = np.argsort(distances)[: self.k]
        KNearestLabel = [self.ytrain[i] for i in kIndices]

        most_common = Counter(KNearestLabel).most_common(1)
        return most_common[0][0]

    def predict(self, Xtest):
        predictions = [self._predict(x) for x in Xtest]
        return np.array(predictions)

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

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    knn = KNN(k=5)
    knn.fit(X_test_std, y_train)

    plot_decision_regions(X_combined_std, y_combined, classifier=knn)

    # X_train = np.array([[2, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9]])
    # y_train = np.array([0, 0, 0, 1, 1, 1])
    # X_test = np.array([[5, 5], [6, 6]])
    # y_test = np.array([0, 1])

    # knn = KNN(k=3)
    # knn.fit(X_train, y_train)

    # #predicitons = knn.predict(X_test)
    # #print('predictions: ', predicitons)
    # plot_decision_regions(X_test, y_test, classifier=knn)

if __name__ == '__main__':
    main()
