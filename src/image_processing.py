from os import listdir, mkdir
from os.path import isfile, join
from shutil import copyfile
import cv2
import numpy as np
import matplotlib.pyplot as plt

def square_crop(image):
    """Crops original image to square

    Parameters
    ----------
    image: numpy array

    Returns
    -------
    shape of cropped image as tuple
    """

    height = image.shape[0]
    width = image.shape[1]

    if height < width:
        cropped = image[:, :height]
    else:
        cropped = image[:width, :]
    return cropped.shape

def resize_img(img_root, target_root, new_size, border_size, border_color):
    """Resizes image, adds border, and changes color channel

    Parameters
    ----------
    img_root: root directory
    target_root: target directory
    new_size: desired size as tuple
    border_size: desired border size as int
    border_color: desired border color

    Returns
    -------
    modified image saved to target directory
    """

    files = [f for f in listdir(img_root) if isfile(join(img_root, f))]

    for file in files:
        img = cv2.imread('{}{}'.format(img_root, file))

        #resize to square and smaller size
        dims = square_crop(img)
        img = cv2.resize(img, (dims[:2]))
        img = cv2.resize(img, new_size)

        #border
        #img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)

        #color channel
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #or HSV

        cv2.imwrite('{}/{}'.format(target_root, file), img)


def train_test_split(target_root, split=.20):
    """Splits data

    Parameters
    ----------
    target_root: target directory
    split: desired split between train and test

    Returns
    -------
    makes train and test directories in target directory and adds random images to train and test directories
    """
    files = np.array([f for f in listdir(target_root) if isfile(join(target_root, f))])

    train_path = target_root + 'train/'
    test_path = target_root + 'test/'
    valid_path = target_root + 'valid/'
    # mkdir(train_path)
    # mkdir(test_path)
    # mkdir(valid_path)

    test_files = np.random.choice(files, int(split*len(files)))
    for file in test_files:
        copyfile(target_root + file, test_path + file)
    train_files = files[~np.in1d(files, test_files)]
    for file in train_files:
        copyfile(target_root + file, train_path + file)

if __name__ == '__main__':
    # img_root = 'images/'
    img_root = '../images/vendors/'
    # target_root = 'thumbnails/'
    target_root = '../thumbnails/test/'

    #train_test_split(target_root) #for some reason have to run this twice commenting out the mkdir
    resize_img(img_root, target_root, new_size=(100, 100), border_size=20, border_color=[0,0,0])
