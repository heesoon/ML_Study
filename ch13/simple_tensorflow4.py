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
    y_train = np.array([1.0, 1.3, 3.1,
                        2.0, 5.0, 6.3,
                        6.6, 7.4, 8.0,
                        9.0])

    X_train_norm = (X_train - np.mean(X_train))/np.std(X_train)

    ds_train_orig = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_train_norm, tf.float32),
        tf.cast(y_train, tf.float32)))

    tf.random.set_seed(1)
    num_epochs = 200
    batch_size = 1
    model = MyModel()
    #model.build((None, 1))

    model.compile(optimizer='sgd', 
                loss=loss_fn,
                metrics=['mae', 'mse'])
    
    model.fit(X_train_norm, y_train, 
            epochs=num_epochs, batch_size=batch_size,
            verbose=1)
 
    print(f'Final weight: {model.w.numpy()}, Final bias: {model.b.numpy()}')

    X_test = np.linspace(0, 9, num=100).reshape(-1, 1)
    X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)

    y_pred = model(tf.cast(X_test_norm, dtype=tf.float32))

    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(1, 2, 1)
    plt.plot(X_train_norm, y_train, 'o', markersize=10)
    plt.plot(X_test_norm, y_pred, '--', lw=3)
    plt.legend(['Training Samples', 'Linear Regression'], fontsize=15)
    plt.show()
    
if __name__ == '__main__':
    main()


