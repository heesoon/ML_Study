import numpy as np
import pandas as pd
import gzip
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from collections import Counter
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

encoder = None

## 단계 3-A: 변환 함수 정의하기
def encode(text_tensor, label):
    global encode
    if encoder is None:
        raise ValueError("Encoder is not initialized.")

    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text, label

## 단계 3-B: 인코딩 함수를 텐서플로 연산으로 감싸기
def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], 
                          Tout=(tf.int64, tf.int64))

def main():

    global encoder 
    with gzip.open('D:\project\python\ML_Study\ch16\movie_data.csv.gz', 'rb') as f:
        df = pd.read_csv(f)
    
    target = df.pop('sentiment')
    ds_raw = tf.data.Dataset.from_tensor_slices((df.values, target.values))
    
    #for ex in ds_raw.take(3):
    #    tf.print(ex[0].numpy()[0][:50], ex[1])
    
    tf.random.set_seed(1)
    ds_raw = ds_raw.shuffle(
        50000, reshuffle_each_iteration=False)
    ds_raw_test = ds_raw.take(25000)
    ds_raw_train_valid = ds_raw.skip(25000)
    ds_raw_train = ds_raw_train_valid.take(20000)
    ds_raw_valid = ds_raw_train_valid.skip(20000)
    
    tokenizer = tfds.deprecated.text.Tokenizer()
    token_counts = Counter()
    #for example in ds_raw_train:
    #    tokens = tokenizer.tokenize(example[0].numpy()[0])
    #    token_counts.update(tokens)
    #print('어휘 사전 크기: ', len(token_counts))
    
    encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)
    ds_train = ds_raw_train.map(encode_map_fn)
    ds_valid = ds_raw_valid.map(encode_map_fn)
    ds_test = ds_raw_test.map(encode_map_fn)

    #tf.random.set_seed(1)
    #for example in ds_train.shuffle(1000).take(5):
    #    print('시퀀스 길이:', example[0].shape)
    
    ## 배치 데이터 만들기
    train_data = ds_train.padded_batch(
        32, padded_shapes=([-1],[]))

    valid_data = ds_valid.padded_batch(
        32, padded_shapes=([-1],[]))

    test_data = ds_test.padded_batch(
        32, padded_shapes=([-1],[]))

    embedding_dim = 20
    vocab_size = len(token_counts) + 2

    tf.random.set_seed(1)

    ## 모델 생성
    bi_lstm_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name='embed-layer'),
        
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, name='lstm-layer'),
            name='bidir-lstm'), 

        tf.keras.layers.Dense(64, activation='relu'),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    ## 컴파일과 훈련:
    bi_lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy'])

    history = bi_lstm_model.fit(
        train_data, 
        validation_data=valid_data, 
        epochs=10)

    ## 테스트 데이터에서 평가
    test_results= bi_lstm_model.evaluate(test_data)
    print('테스트 정확도: {:.2f}%'.format(test_results[1]*100))

if __name__ == '__main__':
    main()