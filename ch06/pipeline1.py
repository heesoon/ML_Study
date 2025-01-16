import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

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
    pipe_lr.fit(X=X_train, y=y_train)
    y_pred = pipe_lr.predict(X_test)

    print('테스트 정확도: %.3f' % pipe_lr.score(X_test, y_test))

if __name__ == '__main__':
    main()