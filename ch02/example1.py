import os
import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
        
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
#s = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data')
#print('URL: ', s)
#df = pd.read_csv(s, header=None, encoding='utf-8')
#df.to_csv("irisdata.csv", index=False)

def main():
    df = pd.read_csv('/Volumes/Macintosh_HD_2/python/ml/chapter1/iris.data',
                 header=None, encoding='utf-8')
    df.tail()
    #print(df)
    
    y = df.iloc[0:100, 4].values
    #print(y)
    y = np.where(y == 'iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    #print(X)
   
    #plt.scatter(X[:50, 0], X[:50, 1],
    #            color='red', marker='o', label='setosa')
    #plt.scatter(X[50:100, 0], X[50:100, 1],
    #            color='blue', marker='x', label='versicolor')
    #plt.xlabel('sepal length [cm]')
    #plt.ylabel('petal length [cm]')
    #plt.legend(loc='upper left')
    #plt.show()   

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.show()

if __name__ == '__main__':
    main()


