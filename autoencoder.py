import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras import backend as K
from os import listdir, mkdir
from os.path import isfile, join
import cv2

img_width, img_height = 98, 98

def autoencoder_model(X_train):
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 1)

    input_img = Input(shape=(X_train.shape[1],))

    # first encoding layer
    encoded1 = Dense(units = 256, activation = 'relu')(input_img)

    # second encoding layer
    # note that each layer is multiplied by the layer before
    encoded2 = Dense(units = 64, activation='relu')(encoded1)

    # first decoding layer
    decoded1 = Dense(units = 256, activation='relu')(encoded2)

    # second decoding layer - this produces the output
    decoded2 = Dense(units = 784, activation='sigmoid')(decoded1)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded2)

    # compile model
    autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mse'])

    return autoencoder

if __name__ == '__main__':
    np.random.seed(13) #stuff from case study - autoencoder (cnn.py)

    batch_size = 16
    target_root = 'thumbnails/'
    # train_datagen = ImageDataGenerator(
    #         rescale=1./255,
    #         shear_range=0.2,
    #         zoom_range=0.2,
    #         horizontal_flip=True)
    #
    # test_datagen = ImageDataGenerator(rescale=1./255)
    #
    # train_generator = train_datagen.flow_from_directory(
    #         'thumbnails/train',
    #         target_size=(28,28),
    #         batch_size=batch_size)
    #
    # test_generator = test_datagen.flow_from_directory(
    #         'thumbnails/test',
    #         target_size=(28,28),
    #         batch_size=batch_size)

    # model = autoencoder_model()
    # model.fit_generator(train_generator,
    #                     steps_per_epoch = 2000 // batch_size,
    #                     epochs = 10,
    #                     validation_data=test_generator,
    #                     validation_steps = 800 // batch_size)

    x = []
    files = [f for f in listdir(target_root) if isfile(join(target_root, f))]
    for file in files:
        img = cv2.imread('{}{}'.format(target_root, file))
        x.append(img)
    x = np.array(x)
