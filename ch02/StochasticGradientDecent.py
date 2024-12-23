import os
import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from matplotlib.colors import ListedColormap

class AdallineSGD(object):
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self._shuffle = shuffle
        self.random_state = random_state

    def initialize_weights(self, X):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

    def update_weights(self, x, target):
        output = self.activation(self.net_input(x))
        error = target - output
        self.w_[1:] += self.eta * x.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def fit(self, X, y):

        self.initialize_weights(X)
        self.cost_ = []

        for i in range(self.n_iter):
            if self._shuffle:
                X, y = self.shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self.update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

def plot_decision_regions_v2(ax, X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors=colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    ax.set(xlim=(xx1.min(), xx1.max()), ylim=(xx2.min(), xx2.max()))

    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolors='black')

    ax.set_xlabel('sepal length [cm]')
    ax.set_ylabel('petal length [cm]')    
    ax.legend(loc='upper left')

def plot_error_number_of_epochs(ax, errors):
    ax.plot(range(1, len(errors) + 1), errors, marker='o')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Average-Error')

def main():

    if os.name == 'posix':
        print('unix family')
        df = pd.read_csv('/Volumes/Macintosh_HD_2/python/ml/chapter1/iris.data',
                    header=None, encoding='utf-8')
    elif os.name == 'nt':
        print('Windows family')
        df = pd.read_csv('D:\Project\ML_Study\ch02\iris.data',
                    header=None, encoding='utf-8')
    else:
        print('Unknown')

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    # Nomalize X charateristics
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean())/X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean())/X[:, 1].std()

    ada = AdallineSGD(n_iter=15, eta=0.01)
    ada.fit(X_std, y)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    plot_decision_regions_v2(ax[0], X_std, y, classifier=ada)
    plot_error_number_of_epochs(ax[1], ada.cost_)
    plt.show()

if __name__ == '__main__':
    main()


