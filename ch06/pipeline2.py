import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

def model_verification1(pipe, X, y):
    kfold = StratifiedKFold(n_splits=10).split(X, y)
    scores = []

    for k, (train, test) in enumerate(kfold):
        pipe.fit(X[train], y[train])
        score = pipe.score(X[test], y[test])
        scores.append(score)
        print('폴드: %2d, 클래스 분포: %s, 정확도: %.3f' %(k+1, np.bincount(y[train]), score))
    
    print('\nCV 정확도: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

def model_verification2(pipe, X, y):
    scores = cross_val_score(estimator=pipe, X=X, y=y, cv=10, n_jobs=-1)
    print('\n CV 정확도 점수: %s' %scores)
    print('\nCV 정확도: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

def main():
    if os.name == 'posix':
        print('unix family')
        df = pd.read_csv('/Volumes/Macintosh_HD_2/python/ml/chapter1/iris.data',
                    header=None, encoding='utf-8')
    elif os.name == 'nt':
        print('Windows family')
        df = pd.read_csv('D:\Project\ML_Study\ch06\wdbc.data',
                    header=None, encoding='utf-8')
    else:
        print('Unknown')

    X = df.loc[:, 2:].values #특성열
    y = df.loc[:, 1].values #결과
    le = LabelEncoder()
    y = le.fit_transform(y) #['M', 'B'] -> [1, 0] 변환
    #print(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

    pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(random_state=1))

    model_verification1(pipe_lr, X_train, y_train)
    model_verification2(pipe_lr, X_train, y_train)

if __name__ == '__main__':
    main()