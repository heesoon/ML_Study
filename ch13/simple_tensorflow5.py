import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Keras에서 가중치로 정의
        self.w = self.add_weight(name='weight', shape=(), initializer='zeros', trainable=True)
        self.b = self.add_weight(name='bias', shape=(), initializer='zeros', trainable=True)

    def call(self, x):
        return self.w * x + self.b

def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def main():
    X_train = np.arange(10).reshape((10, 1))
    y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])

    X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)

    model = MyModel()
    
    tf.random.set_seed(1)
    num_epochs = 200
    batch_size = 1
    learning_rate = 0.01  # 학습률 설정

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), 
                  loss=loss_fn)
    
    # Keras의 model.fit() 사용
    model.fit(X_train_norm, y_train, 
              epochs=num_epochs, batch_size=batch_size, 
              verbose=1)

    # 최종 모델 파라미터 출력
    print(f'Final weight: {model.w.numpy()}, Final bias: {model.b.numpy()}')

if __name__ == '__main__':
    main()
