# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 20:32:37 2018

@author: Harsh Sharma

This script just loads the images and saves them into NumPy 
binary format files .npy for faster loading later.

Inspired from 
https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/data.py

"""

from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread

data_path = 'data/'
train_data_path = os.path.join(data_path, 'train')
test_data_path = os.path.join(data_path, 'test')

image_height = 420 #rows
image_width  = 580 #cols


def create_data(isTrain): 
    data_path = train_data_path if isTrain else test_data_path
    images = os.listdir(data_path) 
    total =  int(len(images)/2) if isTrain else len(images) #base images and mask images

    imgs = np.ndarray((total, image_height, image_width), dtype=np.uint8)
    if (isTrain):
        imgs_mask = np.ndarray((total, image_height, image_width), dtype=np.uint8)
    else:
        imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    for image_name in images:
        if 'mask' in image_name:
            continue
        
        img = imread(os.path.join(data_path, image_name), as_grey=True)
        imgs[i] = np.array([img])
        
        if (isTrain):
            image_mask_name = image_name.split('.')[0] + '_mask.tif'
            img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)
            imgs_mask[i] = np.array([img_mask])
        else:
            img_id = int(image_name.split('.')[0])
            imgs_id[i] = img_id
        
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    
    if (isTrain):
        np.save('imgs_train.npy', imgs)
        np.save('imgs_mask_train.npy', imgs_mask)
    else:
        np.save('imgs_test.npy', imgs)
        np.save('imgs_id_test.npy', imgs_id)
    
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
#    create_data(True)
    create_data(False)