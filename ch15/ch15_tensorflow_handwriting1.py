import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

BUFFER_SIZE = 100000
BATCH_SIZE = 64
NUM_EPOCHS = 20

def main():
    mnist_bldr = tfds.builder('mnist')
    mnist_bldr.download_and_prepare()
    datasets = mnist_bldr.as_dataset(shuffle_files=False)
    mnist_train_orig = datasets['train']
    mnist_test_orig = datasets['test']

    mnist_train = mnist_train_orig.map(
        lambda item: (tf.cast(item['image'], tf.float32)/255.0,
                      tf.cast(item['label'], tf.int32)))

    mnist_test = mnist_test_orig.map(
        lambda item: (tf.cast(item['image'], tf.float32)/255.0,
                      tf.cast(item['label'], tf.int32)))
    
    tf.random.set_seed(1)
    
    mnist_train = mnist_train.shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=False)
    mnist_valid = mnist_train.take(10000).batch(BATCH_SIZE)
    mnist_train = mnist_train.skip(10000).batch(BATCH_SIZE)
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=32, kernel_size=(5, 5),
        strides=(1, 1), padding='same',
        data_format='channels_last',
        name='conv_1', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), name='pool_1'))
    model.add(tf.keras.layers.Conv2D(
        filters=64, kernel_size=(5, 5),
        strides=(1, 1), padding='same',
        name='conv_2', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), name='pool_2'))
    #model.compute_output_shape(input_shape=(16, 28, 28, 1))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=1024, name='fc_1', activation='relu'))
    model.add(tf.keras.layers.Dropout(
        rate=0.5))
    model.add(tf.keras.layers.Dense(
        units=10, name='fc_2', activation='softmax'))
    
    model.build(input_shape=(None, 28, 28, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
    
    history = model.fit(mnist_train, epochs=NUM_EPOCHS, 
                        validation_data=mnist_valid, shuffle=True)
    
    hist = history.history
    x_arr = np.arange(len(hist['loss'])) + 1

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, hist['loss'], '-o', label='Train loss')
    ax.plot(x_arr, hist['val_loss'], '--<', label='Validation loss')
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)
    ax.legend(fontsize=15)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, hist['accuracy'], '-o', label='Train acc.')
    ax.plot(x_arr, hist['val_accuracy'], '--<', label='Validation acc.')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)
    
    test_results = model.evaluate(mnist_test.batch(20))
    print('\n테스트 정확도 {:.2f}%'.format(test_results[1]*100))

    #plt.savefig('images/15_12.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()