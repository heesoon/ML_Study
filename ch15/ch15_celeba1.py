import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
#https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM

def main():
    #celeba_bldr = tfds.builder('celeb_a')
    #celeba_bldr.download_and_prepare()
    #celeba = celeba_bldr.as_dataset(shuffle_files=False)
    
    celeba_data = tfds.load('celeb_a', data_dir='C:\\Users\\hskim\\tensorflow_datasets\\celeb_a', with_info=True)
    
if __name__ == '__main__':
    main()