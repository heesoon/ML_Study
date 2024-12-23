import os
import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from matplotlib.colors import ListedColormap

class AdallineGD(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
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
    ax.set_ylabel('Sum-Squared-Error')

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

    ada = AdallineGD(n_iter=15, eta=0.01)
    ada.fit(X_std, y)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    plot_decision_regions_v2(ax[0], X_std, y, classifier=ada)
    plot_error_number_of_epochs(ax[1], ada.cost_)

    plt.show()

if __name__ == '__main__':
    main()