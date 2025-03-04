import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

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

    ds_test = ds_test.map(
        lambda x: (x['features'], x['label'])
    )
    
    iris_model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='sigmoid', name='fc1', input_shape=(4,)),
        tf.keras.layers.Dense(3, activation='softmax', name='fc2')
    ])
    
    iris_model.summary()
    
    iris_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    num_epochs = 100
    training_size = 100
    batch_size = 2
    steps_per_epoch = int(np.ceil(training_size / batch_size))

    ds_train = ds_train_orig.shuffle(buffer_size=training_size)
    ds_train = ds_train.repeat()
    ds_train = ds_train.batch(batch_size=batch_size)
    ds_train = ds_train.prefetch(buffer_size=1000)

    history = iris_model.fit(ds_train, epochs=num_epochs,
                            steps_per_epoch=steps_per_epoch, 
                            verbose=0)
    
    hist = history.history

    results = iris_model.evaluate(ds_test.batch(50), verbose=0)
    print('테스트 손실: {:.4f}   테스트 정확도: {:.4f}'.format(*results))

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(hist['loss'], lw=3)
    ax.set_title('Training loss', size=15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(hist['accuracy'], lw=3)
    ax.set_title('Training accuracy', size=15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    # plt.savefig('images/13_7.png', dpi=300)
    plt.show()
    
    iris_model.save('iris-classifier.h5', 
                    overwrite=True,
                    include_optimizer=True,
                    save_format='h5')

if __name__ == '__main__':
    main()
