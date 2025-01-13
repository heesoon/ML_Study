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
        df = pd.read_csv('D:\Project\ML_Study\ch05\wine.data',
                    header=None, encoding='utf-8')
    else:
        print('Unknown')

    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    print("\n교유값: \n%s", eigen_vals)

    plot_explained_variance_ratio(eigen_vals)

if __name__ == '__main__':
    main()