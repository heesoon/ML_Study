import os
import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore

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
    df.tail()

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ada1 = AdallineSGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-Square-Error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    ada2 = AdallineSGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(Sum-Square-Error)')
    ax[1].set_title('Adaline - Learning rate 0.0001')

    plt.show()

if __name__ == '__main__':
    main()


