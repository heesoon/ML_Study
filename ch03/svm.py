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
        self.C = C

    def init_weight(self, X):
        self.w = np.zeros(X.shape[1]) # number of characteristic
        self.b = 0

    def update_weight(self, dw, db):
        self.w -= self.eta * dw
        self.b -= self.eta * db

    def hinge_loss(self, X, y):
        return 1 - y * (np.dot(X, self.w) + self.b)
    
    def calc_margin(self, X, y):
        return y * (np.dot(X, self.w) + self.b)
    
    def activation(self, z):
        return z

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def fit(self, X, y):
        self.init_weight(X)
        for i in range(self.n_iter):
            dw = np.zeros(X.shape[1])
            db = 0
            
            for idx, xi in enumerate(X):
                if self.hinge_loss(xi, y[idx]) > 0:
                    dw += self.C * y[idx]*xi
                    db += self.C * y[idx]
                #if self.calc_margin(xi, y[idx]) < 1:
                #     dw += self.C * y[idx]*xi
                #     db += self.C * y[idx]

            self.update_weight(dw, db)
            
        print('optimum w :', self.w)
        print('optimum b :', self.b)
        return self

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
    # 1. Iris 데이터셋 로드
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]  # 데이터(특성)
    y = iris.target  # 레이블

    # 2. y == 0과 y == 1인 샘플만 선택
    mask = (y == 0) | (y == 1)  # y가 0 또는 1인 샘플만 선택
    X_filtered = X[mask]  # 필터링된 특성
    y_filtered = y[mask]  # 필터링된 레이블

    # 3. 학습과 테스트 데이터로 나누기
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.3, random_state=42)

    # 4. 데이터 표준화
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    sc = StandardScaler()
    sc.fit(X_train)
    
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    svm = SVM(eta=0.001, n_iter=1000, C=1)
    svm.fit(X_train_std, y_train)
    plot_decision_regions(X=X_test_std, y=y_test, classifier=svm)

if __name__ == '__main__':
    main()