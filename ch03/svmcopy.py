import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

class SVM:
    def __init__(self, C=1.0, learning_rate=0.01, epochs=1000):
        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        m, n = X.shape  # m은 샘플 수, n은 특성 수
        self.w = np.zeros(n)  # w는 특성 수만큼 초기화
        self.b = 0
        
        # Gradient Descent
        for epoch in range(self.epochs):
            dw = np.zeros(n)
            db = 0
            for i in range(m):
                # SVM의 힌지 손실 조건
                condition = y[i] * (np.dot(X[i], self.w) + self.b) >= 1
                if condition:
                    dw += self.w  # Regularization term
                else:
                    dw += self.w - self.C * y[i] * X[i]  # Misclassified points
                    db += -self.C * y[i]  # Misclassified points
            
            # Update weights and bias
            self.w -= self.learning_rate * dw / m
            self.b -= self.learning_rate * db / m

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                loss = self.compute_loss(X, y)
                print(f'Epoch {epoch}: Loss = {loss}')
    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
    
    def compute_loss(self, X, y):
        m = X.shape[0]
        # Hinge loss 계산
        hinge_loss = np.mean(np.maximum(0, 1 - y * (np.dot(X, self.w) + self.b)))
        # Regularization term
        regularization_loss = 0.5 * np.dot(self.w, self.w)
        return regularization_loss + self.C * hinge_loss


# 예제 데이터 생성
n_features = 2  # 특성 수를 2로 설정
X, y = make_classification(n_samples=100, n_features=n_features, n_classes=2, 
                            n_informative=2, n_redundant=0, n_repeated=0, random_state=42)

y = 2 * y - 1  # y를 -1, 1로 변환

# SVM 모델 학습
svm = SVM(C=1.0, learning_rate=0.01, epochs=1000)
svm.fit(X, y)

# 예측
y_pred = svm.predict(X)

# 결과 시각화
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', marker='o')
plt.title('SVM with Gradient Descent')

# 경계선 그리기
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100), 
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

plt.show()
