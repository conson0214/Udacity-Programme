# -*- coding: utf-8 -*-
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

import h5py
import datetime

import numpy as np

import cv2
import os
from tqdm import tqdm
import math


dog_train_path = './Dataset/train/dog'
cat_train_path = './Dataset/train/cat'
test_path = './Dataset/test2'


def abnormal_data_reject():
    dog_train_list = os.listdir(dog_train_path)
    cat_train_list = os.listdir(cat_train_path)
    with open('./abnormal.txt', 'r') as f:
        while 1:
            line = f.readline()
            if not line:
                break
            img_name = line.replace('\n', '')
            if 'dog' in line:
                dog_train_list.remove(img_name)
            else:
                cat_train_list.remove(img_name)
    dog_train_list = [os.path.join(dog_train_path, f) for f in dog_train_list]
    cat_train_list = [os.path.join(cat_train_path, f) for f in cat_train_list]
    train_list = dog_train_list + cat_train_list

    return train_list


def write_gap(MODEL, image_size, lambda_func=None):
    # get model
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)

    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    # data generate
    n = all_train_list.__len__()
    X = np.zeros((n, image_size[0], image_size[1], 3), dtype=np.uint8)
    y = np.zeros((n, 1), dtype=np.uint8)
    X_file = 'X' + str(image_size[0]) + '.npy'
    y_file = 'y' + str(image_size[0]) + '.npy'

    if os.path.exists(X_file) and os.path.exists(y_file):
        X = np.load(X_file)
        y = np.load(y_file)
    else:
        for idx, img_name in tqdm(enumerate(all_train_list)):
            X[idx] = cv2.resize(cv2.imread(img_name), (image_size[0], image_size[1]))
            if 'dog' in img_name:
                y[idx] = 1
            else:
                y[idx] = 0
        np.save(X_file, X)
        np.save(y_file, y)

    train_data_gen = ImageDataGenerator()
    test_data_gen = ImageDataGenerator()

    train_generator = train_data_gen.flow(X, y, shuffle=False, batch_size=100)
    test_generator = test_data_gen.flow_from_directory("/home/autel/Dataset/Cat_vs_Dog/test2", image_size, shuffle=False
                                                       , batch_size=100, class_mode=None)
    # feature get
    train = model.predict_generator(train_generator, math.ceil(n/100))
    test = model.predict_generator(test_generator, int(test_generator.samples/100))
    with h5py.File("gap_%s.h5" % MODEL.__name__) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=y)


all_train_list = abnormal_data_reject()

starttime = datetime.datetime.now()
write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)
write_gap(Xception, (299, 299), xception.preprocess_input)
write_gap(InceptionResNetV2, (299, 299), inception_resnet_v2.preprocess_input)
endtime = datetime.datetime.now()
print((endtime-starttime).seconds)
