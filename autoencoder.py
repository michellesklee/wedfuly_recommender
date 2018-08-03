import cv2
import glob
from PIL import Image
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, UpSampling2D, Convolution2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.utils import np_utils, layer_utils
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import plot_model

def cnn_autoencoder(input_img):
    """CNN Autoencoder

    Parameters
    ----------
    input_img: images as numpy arrays

    Returns
    -------
    CNN Autoencoder model
    """

    #encoder
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #100 100 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 50 50 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #50 50 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # 25 25 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #25 25 128
    encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(conv3) #13 13 128
    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #25 25 128
    up1 = UpSampling2D((2,2))(conv4) #50 50 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) #50 50 64
    up2 = UpSampling2D((2,2))(conv5) #100 100 64
    decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(up2) # 100 100 3

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return autoencoder

def get_encoded(model, x):
    """Runs initial encoder layers of the model

    Parameters
    ----------
    model: neural network with convolutional layer
    x: data to be encoded

    Returns
    -------
    encoded sample as numpy array with dimensions up to last conv layer in encoder (e.g., (386, 25, 25, 128))
    """

    get_encoded = K.function([model.layers[0].input], [model.layers[5].output])
    encoded_sample = get_encoded([x])[0]
    return encoded_sample

def get_batches(x, batch_size=1000):
    '''Split data into batches for large datasets

    Parameters
    ----------
    x: data to be split
    batch_size: desired batch size

    Returns
    -------
    Batches of desired size
    '''

    if len(x) < batch_size:
        return [x]
    n_batches = len(x) // batch_size

    # If batches fit exactly into the size of x.
    if len(x) % batch_size == 0:
        return [x[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]

    # If there is a remainder.
    else:
        return [x[i*batch_size:min((i+1)*batch_size, len(x))] for i in range(n_batches+1)]

def pool_conv_layer(model, x, last_conv_layer=4):
    ''' Pool the model up to the last convolutional layer in encoder

    Parameters
    ----------
    model: neural network with convolutional layer
    x: data to be pooled
    last_conv_layer: last convolutional layer in encoder

    Returns
    -------
    Pooled array with dimensions up to last conv layer in encoder (e.g., (386, 25, 25))
    '''

    get_encoded = K.function([model.layers[0].input],
                             [model.layers[last_conv_layer].output])
    encoded_sample = get_encoded([x])[0]
    pooled_array = encoded_sample.max(axis=-1)
    return pooled_array

def pool_encoded(model, x):
    '''Collect encoded data in batches when datasets are large

    Parameters
    ----------
    model: neural network with convolutional layer
    x: data to be encoded and pooled

    Returns
    -------
    Pooled and encoded data in desired batch saved as .npy file
    '''

    X_encoded = []
    i=0

    for batch in get_batches(x, batch_size=1000):
        i+=1
        print('Running batch... {}'.format(i*len(batch)))
        #runs pooling function on the model for each batch.
        X_encoded.append(pool_conv_layer(model, x_train))

    X_encoded = np.concatenate(X_encoded)
    #np.save('X_encoded.npy', X_encoded)
    return X_encoded

def encoded_compressed(model, x):
    '''Keep layer information (don't pool) and compress

    Parameters
    ----------
    model: neural network with convolutional layer
    x: data to be encoded and pooled

    Returns
    -------
    Encoded and compressed images saved as .npy file
    '''

    X_encoded = pool_encoded(model, x)
    X_encoded_compressed = []
    i=0
    for batch in get_batches(x, batch_size=1000):
        i+=1
        print('Running batch... {}'.format(i*len(batch)))
        encoded_array = get_encoded(model, x)
        X_encoded_compressed.append(encoded_array)

    X_encoded_compressed = np.concatenate(X_encoded_compressed)
    np.save('X_encoded_compressed.npy', X_encoded_compressed)

    # Reshape the arrays so we can run them though a clustering algorithm.
    X_encoded_reshape = X_encoded.reshape(X_encoded.shape[0],
                                          X_encoded.shape[1]*X_encoded.shape[2])
    print('Encoded shape:', X_encoded_reshape.shape) #(317, 625)

    # Slightly different code when we're working with the smaller model dimensions.
    X_encoded_compressed_reshape = X_encoded_compressed.reshape(X_encoded_compressed.shape[0],
                                                                X_encoded_compressed.shape[1]*X_encoded_compressed.shape[2]*X_encoded_compressed.shape[3])
    print('Encoded compressed shape:', X_encoded_compressed_reshape.shape)
    return X_encoded_compressed_reshape


########### VISUALIZATION ###########

def plot_reconstruction(x_orig, x_decoded, n=10, plotname=None):
    '''Visualize images before and after running through autoencoder

    Parameters
    ----------
    x_orig: 2D numpy array
    x_recon: 2D numpy array
    n: int, number of images to plot
    plotname: str, path to save file

    Returns
    -------
    Plot of original and decoded images
    '''

    plt.figure(figsize=(n*2, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_orig[i].reshape(100, 100, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(X_decoded[i].reshape(100, 100, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if plotname:
        plt.savefig(plotname)
    else:
        plt.show()

def get_encoded_plot(model, x):
    """Visualize the model after pooling up to last convolutional layer to see where the model is paying most attention

    Parameters
    ----------
    model: neural network with convolutional layer
    x: data to be encoded

    Returns
    -------
    Plot of original image and encoded image
    Encoded image shows the max of the filters (i.e., axis=-1)
    """

    get_encoded = K.function([model.layers[0].input], [model.layers[5].output])
    encoded_sample = get_encoded([x])[0] #should have shape like (25, 25, 128)

    for n_image in [1, 3, 7]:
        plt.figure(figsize=(18,4))

        plt.subplot(1,4,1)
        plt.imshow(x[n_image][:,:,::-1])
        plt.axis('off')
        plt.title('Original Image')

        plt.subplot(1,4,2)
        plt.imshow(encoded_sample[n_image].max(axis=-1)) #axis=-1 == 128
        plt.axis('off')
        plt.title('Encoded Max')

        plt.show()

def attention_model(model, x, n_images, last_conv_layer=4):
    """Visualize model with pooled filters up to the last convolutional layer in encoder to see where the model is paying most attention

    Parameters
    ----------
    model: neural network with convolutional layer
    x: data to be encoded
    n_images: desired number of images
    last_conv_layer: last convolutional layer in encoder

    Returns
    -------
    Plot of original image and images up to the last convolutional layer
    Encoded image shows the max of the filters (i.e., axis=-1)
    """
    # Randomly choose the arrays to plot.
    X_to_plot = x[np.random.choice(range(len(x)), 2*n_images, replace=False)]
    # Run pool_conv_layer to get the pooled image.
    pooled_array = pool_conv_layer(model, X_to_plot, last_conv_layer)

    # Plot them,
    plt.figure(figsize=(14, 4*n_images))
    for i in range(n_images):
        plt.subplot(n_images, 4, 4*i+1)
        plt.imshow(X_to_plot[i][:,:,::-1])
        plt.axis('off')
        plt.subplot(n_images, 4, 4*i+2)
        # Resize the encoded image to be the same as the original.
        plt.imshow(cv2.resize(pooled_array[i], (X_to_plot.shape[1], X_to_plot.shape[2])))
        plt.axis('off')
        # four per row.
        plt.subplot(n_images, 4, 4*i+3)
        plt.imshow(X_to_plot[i+n_images][:,:,::-1])
        plt.axis('off')
        plt.subplot(n_images, 4, 4*i+4)
        plt.imshow(cv2.resize(pooled_array[i+n_images], (X_to_plot.shape[1], X_to_plot.shape[2])))
        plt.axis('off')

    plt.show()


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

    #cnn model
    cnn_model = cnn_autoencoder(input_img)
    cnn_model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1, validation_split=0.1)
    restored_imgs = cnn_model.predict(x_test)
    cnn_model.save('cnn_model1.h5')

    #plot cnn architecture
    plot_model(cnn_model, show_shapes=True, to_file = 'model1.png')

    #plot reconstructed images
    plot_reconstruction(x_test, restored_imgs, n=10)

    #plot encoded images
    get_encoded_plot(cnn_model, x_train)

    #plot attention model
    attention_model(cnn_model, x_train, 3)

    #move to kmeans
    #X_encoded_compressed_reshape = encoded_compressed(cnn_model, x_train)
    #kmeans(X_encoded_compressed_reshape)
