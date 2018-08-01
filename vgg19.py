import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, UpSampling2D, Convolution2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.models import Model
from keras.utils import np_utils, layer_utils
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils.data_utils import get_file
from os import listdir, mkdir
from os.path import isfile, join
import cv2
import glob
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

def vgg19_model(weights_path, input_img):
    #Block 1
    b1_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img) #100 100 64
    b1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(b1_conv1) #100 100 64
    b1_pool = MaxPooling2D((2, 2), strides=(2, 2))(b1_conv2) #50 50 64

    # Block 2
    b2_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(b1_pool) #50 50 128
    b2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(b2_conv1)#50 50 128
    b2_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(b2_conv2) #25 25 128

    # Block 3
    b3_conv1 = Conv2D(256, (3, 3), activation='relu', padding='same')(b2_pool) #25 25 256
    b3_conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(b3_conv1)#25 25 256
    b3_conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(b3_conv2)#25 25 256
    b3_conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(b3_conv3)#25 25 256
    b3_pool = MaxPooling2D((2, 2), strides=(2, 2))(b3_conv4) #12 12 256

    # Block 4
    b4_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same')(b3_pool) #12 12 512
    b4_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(b4_conv1)
    b4_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(b4_conv2)
    b4_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(b4_conv3)
    b4_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(b4_conv4) #6 6 512

    # Block 5
    b5_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same')(b4_pool)
    b5_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same')(b5_conv1)
    b5_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same')(b5_conv2)
    b5_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(b5_conv3)
    b5_pool = MaxPooling2D((2, 2), strides=(2, 2))(b5_conv4)

    model = Model(input_img, b5_pool)
    return model

if __name__ == '__main__':
    train_files = glob.glob('thumbnails/train/*.jpg')
    x_train = np.array([np.array(Image.open(fname)) for fname in train_files])
    x_train = x_train.reshape(-1, 100, 100, 3)
    x_train = x_train / np.max(x_train)

    test_files = glob.glob('thumbnails/test/*.jpg')
    x_test = np.array([np.array(Image.open(fname)) for fname in test_files])
    x_test = x_test.reshape(-1, 100, 100, 3)
    x_test = x_test / np.max(x_test)

    batch_size = 128
    epochs = 10
    inChannel = 3
    x, y = 100, 100
    input_img = Input(shape = (x, y, inChannel))
    input_shape = (100, 100, 3)

    #vgg model
    weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
    cache_subdir='models')
    vgg_model = vgg19_model(weights_path, input_img)
    vgg_model.compile(optimizer='adam', loss='mean_squared_error')
    #plot_model(vgg_model, show_shapes=True, to_file='vgg_model.png')
    vgg_feature = vgg_model.predict(x_test)
    #vgg_feature.shape = 69, 3, 3, 512
