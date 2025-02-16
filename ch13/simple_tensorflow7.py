import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def main():
    iris, iris_info = tfds.load('iris', with_info=True)
    #print(iris_info)
    
    tf.random.set_seed(1)
    ds_orig = iris['train']
    ds_orig = ds_orig.shuffle(150, reshuffle_each_iteration=False)
    ds_train_orig = ds_orig.take(100)
    ds_test = ds_orig.skip(100)
    
    ds_train_orig = ds_train_orig.map(
        lambda x: (x['features'], x['label'])
    )

    num_epochs = 100
    training_size = 100
    batch_size = 2
    steps_per_epoch = int(np.ceil(training_size / batch_size))

    ds_train = ds_train_orig.shuffle(buffer_size=training_size)
    ds_train = ds_train.repeat()
    ds_train = ds_train.batch(batch_size=batch_size)
    ds_train = ds_train.prefetch(buffer_size=1000)

    ds_test = ds_test.map(
        lambda x: (x['features'], x['label'])
    )
    
    iris_model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='sigmoid', name='fc1', input_shape=(4,)),
        tf.keras.layers.Dense(3, activation='softmax', name='fc2')
    ])

    callback_list = [ModelCheckpoint(filepath='iris-earlystopping.h5', 
                                    monitor='val_loss'),
                    EarlyStopping(patience=3, restore_best_weights=True)]

    tf.random.set_seed(1)

    model = tf.keras.models.model_from_json(iris_model.to_json())
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    history = model.fit(ds_train, epochs=500, 
                        steps_per_epoch=steps_per_epoch, 
                        validation_data=ds_test.batch(50), 
                        callbacks=callback_list,
                        verbose=0)
    
    callback_list[1].stopped_epoch
    print(callback_list[1].stopped_epoch)

    hist = history.history

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(hist['loss'], lw=3)
    ax.plot(hist['val_loss'], lw=3)
    ax.set_title('Loss', size=15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(hist['accuracy'], lw=3)
    ax.plot(hist['val_accuracy'], lw=3)
    ax.set_title('Accuracy', size=15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    # plt.savefig('images/13_earlystopping.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
