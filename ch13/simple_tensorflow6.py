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
    
    iris_model_new = tf.keras.models.load_model('iris-classifier.h5')
    iris_model_new.summary()

    results = iris_model_new.evaluate(ds_test.batch(50), verbose=0)
    print('테스트 손실: {:.4f}   테스트 정확도: {:.4f}'.format(*results))
    
if __name__ == '__main__':
    main()
