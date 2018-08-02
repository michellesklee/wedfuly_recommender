from autoencoder import cnn_autoencoder, plot_reconstruction
import numpy as np
import pandas as pd
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
from PIL import Image
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans


def plot_three(im_list):
    plt.figure(figsize=(14,4))
    for i, array in enumerate(im_list):
        plt.subplot(1, len(im_list), i+1)
        plt.imshow(array[:,:,::-1])
        plt.axis('off')
    plt.show()

def elbow_plot(x):
    cluster_errors = []

    K = range(1, 10)
    for k in K:
        km = KMeans(n_clusters=k)
        clusters = km.fit(x)
        cluster_errors.append(clusters.inertia_)
    clusters_df = pd.DataFrame({"num_clusters":K, "cluster_errors": cluster_errors})
    plt.figure(figsize=(12,6))
    plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker="o")
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title('Elbow Plot for Optimal k')
    #plt.savefig('elbow_plot.png')
    plt.show()


def plot_images_from_cluster(x, n_clusters):
    labels = kmeans_encoded(x, n_clusters)
    for i in range(n_clusters):
        print('Cluster {}: {} Elements'.format(i, (labels==i).sum()))
        plot_three(x[np.random.choice(np.where(labels==i)[0], 9, replace=False), :])

def choose_three(x, labels, clusters):
    for i in range(clusters):
        plot_three(x[np.random.choice(np.where(labels==i)[0], 9, replace=False), :])

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

    cnn_model = cnn_autoencoder(input_img)
    cnn_model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1, validation_split=0.1)
    restored_imgs = cnn_model.predict(x_test)
    plot_reconstruction(x_test, restored_imgs, n=10)
    #plot_model(cnn_model, show_shapes=True, to_file = 'model2.png')
    #cnn_model.save('cnn_model2.h5')

    n_clusters = 7
    train_tall = x_train.reshape(np.size(x_train, 0), x_train[0].size)
    test_tall = x_test.reshape(np.size(x_test, 0), x_test[0].size)

    km = KMeans(n_clusters=n_clusters)
    km.fit(train_tall)
    test_labels = km.predict(test_tall)
    train_labels = km.labels_

    print("Test labels: {}".format(test_labels))
    print("Train labels: {}".format(train_labels))

    train_with_labels = np.append(train_tall, train_labels.reshape(np.size(train_labels,0), 1), axis=1)
    test_with_labels = np.append(test_tall, test_labels.reshape(np.size(test_labels,0), 1), axis=1)

    #plot three from cluster
    choose_three(x_test, test_labels, 7)

    ###### VISUALIZATION ######
    elbow_plot(train_tall)

    #cluster histogram
    plt.figure(figsize=(10,4))
    plt.hist(train_labels, bins=n_clusters)
    plt.xticks(range(n_clusters))
    plt.xlabel('k')
    plt.ylabel('number of images')
    plt.title('Number of Images Per Cluster')
    plt.savefig('figures/cluster_hist.png')
    plt.show()

    # Display nine images from each cluster.
    for i in range(n_clusters):
        print('Cluster {}: {} Elements'.format(i, (km.labels_==i).sum()))
        plot_three(x_train[np.random.choice(np.where(km.labels_==i)[0], 9, replace=False), :])

    #plot three images from each cluster
    plot_images_from_cluster(x_train, n_clusters=7)
