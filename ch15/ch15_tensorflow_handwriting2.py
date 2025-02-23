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
    
    batch_test = next(iter(mnist_test.batch(12)))
    preds = model(batch_test[0])
    preds = tf.argmax(preds, axis=1)

    fig = plt.figure(figsize=(12, 4))
    for i in range(12):
        ax = fig.add_subplot(2, 6, i+1)
        ax.set_xticks([]); ax.set_yticks([])
        img = batch_test[0][i, :, :, 0]
        ax.imshow(img, cmap='gray_r')
        ax.text(0.9, 0.1, '{}'.format(preds[i]), 
                size=15, color='blue',
                horizontalalignment='center',
                verticalalignment='center', 
                transform=ax.transAxes)
        
    #plt.savefig('images/15_13.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()