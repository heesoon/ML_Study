import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def plot_explained_variance_ratio(eigen_vals):
    tot = sum(eigen_vals)

    val_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
    cum_val_exp = np.cumsum(val_exp)

    plt.bar(range(1, len(eigen_vals)+1), val_exp, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1, len(eigen_vals)+1), cum_val_exp, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def main():
    if os.name == 'posix':
        print('unix family')
        df = pd.read_csv('/Volumes/Macintosh_HD_2/python/ml/chapter1/iris.data',
                    header=None, encoding='utf-8')
    elif os.name == 'nt':
        print('Windows family')
        df = pd.read_csv('D:\project\python\ML_Study\ch05\wine.data',
                    header=None, encoding='utf-8')
    else:
        print('Unknown')

    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    np.set_printoptions(precision=4)
    
    mean_vecs = []
    for label in range(1, 4):
        mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
        print('MV %s:%s\n' %(label, mean_vecs[label-1]))
        
    d = 13
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.zeros((d, d))
        for row in X_train_std[y_train == label]:
            row, mv = row.reshape(d, 1), mv.reshape(d, 1)
            class_scatter += (row - mv).dot((row - mv).T)
        S_W += class_scatter
    print('Scattering in Class: %sx%s' %(S_W.shape[0], S_W.shape[1]))
    
    print('Class Label Distribution: %s' %np.bincount(y_train)[1:])
    
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.cov(X_train_std[y_train==label].T)
        S_W += class_scatter
    
    mean_overall = np.mean(X_test_std, axis=0)
    mean_overall = mean_overall.reshape(d, 1)
    
    S_B = np.zeros((d, d))
    for i, mean_vec in enumerate(mean_vecs):
        n = X_train[y_train == i + 1, :].shape[0]
        mean_vec = mean_vec.reshape(d, 1)
        S_B += n*(mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

    print('Scattering between Class: %sx%s' %(S_B.shape[0], S_B.shape[1]))
    
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    
    eigen_pairs = [(np.abs(eigen_vals[i], eigen_vecs[:,i])
                    ) for i in range(len(eigen_vals))]
    eigen_pairs = sorted(eigen_pairs, key=lambda k:k[0], reverse=True)
    
    for eigen_val in eigen_pairs:
        print(eigen_val[0])
    
    plot_explained_variance_ratio(eigen_vals)

if __name__ == '__main__':
    main()