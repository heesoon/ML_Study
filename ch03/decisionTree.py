import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class DecisionTree(object):
    def __init__(self, maxDepth=None, minSampleSplit=2):
        self.maxDepth = maxDepth
        self.minSampleSplit = minSampleSplit
        self.tree = None

    def calc_gini_impurity(self, leftY, rightY):
        def gini_impurity(y):
            classCounts = np.bincount(y)
            probabilities = classCounts / len(y)
            return 1 - np.sum(probabilities ** 2)
        
        leftGini = gini_impurity(leftY)
        rightGini = gini_impurity(rightY)

        leftWeight = len(leftY) / (len(leftY) + len(rightY))
        rightWeight = len(rightY) / (len(leftY) + len(rightY))

        return leftWeight * leftGini + rightWeight * rightGini
    
    def build_tree(self, X, y, depth=0):
        # 트리의 깊이가 max_depth에 도달하거나, 최소 샘플 수보다 적으면 리프 노드
        if len(np.unique(y)) == 1:  # 모든 레이블이 같으면 리프 노드
            return {'leaf': True, 'class': y[0]}
        if len(y) < self.minSampleSplit:  # 최소 샘플 수 이하
            return {'leaf': True, 'class': np.bincount(y).argmax()}

        # 트리의 최대 깊이에 도달하면 리프 노드
        if self.maxDepth is not None and depth >= self.maxDepth:
            return {'leaf': True, 'class': np.bincount(y).argmax()}
        
        # 최적의 분할 찾기
        best_gini = float('inf')
        best_split = None
        best_left_y = None
        best_right_y = None
        best_left_X = None
        best_right_X = None

        n_samples, n_features = X.shape
        for feature_idx in range(n_features):  # 각 특성에 대해 반복
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                # 데이터 분할
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                left_X, right_X = X[left_mask], X[right_mask]
                left_y, right_y = y[left_mask], y[right_mask]

                # 분할이 유효한지 확인
                if len(left_y) < self.minSampleSplit or len(right_y) < self.minSampleSplit:
                    continue

                # 지니 불순도 계산
                gini = self.calc_gini_impurity(left_y, right_y)
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_idx, threshold)
                    best_left_X, best_right_X = left_X, right_X
                    best_left_y, best_right_y = left_y, right_y

        if best_split is None:
            return {'leaf': True, 'class': np.bincount(y).argmax()}

        left_tree = self.build_tree(best_left_X, best_left_y, depth + 1)
        right_tree = self.build_tree(best_right_X, best_right_y, depth + 1)

        return {
            'leaf': False,
            'feature_idx': best_split[0],
            'threshold': best_split[1],
            'left': left_tree,
            'right': right_tree
        }

    def predict_single(self, x, tree):
        if tree['leaf']:
            return tree['class']
        if x[tree['feature_idx']] <= tree['threshold']:
            return self.predict_single(x, tree['left'])
        else:
            return self.predict_single(x, tree['right'])

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict(self, X):
        return [self.predict_single(x, self.tree) for x in X]

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors=colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = np.array(Z).reshape(xx1.shape)

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

    dt = DecisionTree()
    dt.fit(X_train_std, y_train)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(X_combined_std, y_combined, classifier=dt)

    # X = np.array([[2, 3], [10, 15], [2, 5], [10, 10], [5, 8], [7, 12]])
    # y = np.array([0, 1, 0, 1, 0, 1])
    # dt = DecisionTree(maxDepth=3)
    # dt.fit(X, y)
    # print(dt.tree)
    # plot_decision_regions(X, y, classifier=dt)

if __name__ == '__main__':
    main()
