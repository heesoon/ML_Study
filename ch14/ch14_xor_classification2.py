import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
    
def plot_performance(hist, x_valid, y_valid, model):
    history = hist.history

    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 3, 1)
    plt.plot(history['loss'], lw=4)
    plt.plot(history['val_loss'], lw=4)
    plt.legend(['Train loss', 'Validation loss'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)

    ax = fig.add_subplot(1, 3, 2)
    plt.plot(history['binary_accuracy'], lw=4)
    plt.plot(history['val_binary_accuracy'], lw=4)
    plt.legend(['Train Acc.', 'Validation Acc.'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)

    ax = fig.add_subplot(1, 3, 3)
    plot_decision_regions(X=x_valid, y=y_valid.astype(np.integer),
                        clf=model)
    ax.set_xlabel(r'$x_1$', size=15)
    ax.xaxis.set_label_coords(1, -0.025)
    ax.set_ylabel(r'$x_2$', size=15)
    ax.yaxis.set_label_coords(-0.025, 1)
    # plt.savefig('images/14_2.png', dpi=300)
    plt.show()

def main():
    tf.random.set_seed(1)
    np.random.seed(1)
    
    x = np.random.uniform(low=-1, high=1, size=(200, 2))
    y = np.ones(len(x))
    y[x[:, 0] * x[:, 1] < 0] = 0
    
    x_train = x[:100, :]
    y_train = y[:100]
    
    x_valid = x[100:, :]
    y_valid = y[100:]
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=4, 
                                    input_shape=(2,), 
                                    activation='relu'))
    model.add(tf.keras.layers.Dense(units=4, activation='relu'))
    model.add(tf.keras.layers.Dense(units=4, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    #model.summary()
    
    model.compile(optimizer=tf.keras.optimizers.SGD(), 
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    
    hist = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), 
                     epochs=200, batch_size=2, verbose=0)
    
    plot_performance(hist, x_valid, y_valid, model)

if __name__ == '__main__':
    main()