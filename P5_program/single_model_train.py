# -*- coding: utf-8 -*-
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.callbacks import *

import numpy as np
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split

import math
import os

import datetime

dog_train_path = './Dataset/train/dog'
cat_train_path = './Dataset/train/cat'


def check_path_gen():
    if not(os.path.exists('./Weights')):
        os.mkdir('Weights')
    if not(os.path.exists('./Tensor_log')):
        os.mkdir('Tensor_log')


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


def train_cat_dog(MODEL, image_size, preprocess_func):
    # hyper-parameters
    bs = 16

    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)

    # 搭建tranfer learning的最后一层
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.25)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(base_model.input, x)
    model.compile(optimizer='adadelta',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

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

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)

    train_generator = train_datagen.flow(X_train, y_train, batch_size=bs)
    val_generator = val_datagen.flow(X_valid, y_valid, batch_size=bs)

    # callbacks
    best_weights_path = os.path.join('./Weights', MODEL.__name__)
    if not(os.path.exists(best_weights_path)):
        os.mkdir(best_weights_path)
    best_weights_filepath = os.path.join(best_weights_path, 'best_weights')

    log_path = os.path.join('./Tensor_log', MODEL.__name__)
    if not (os.path.exists(log_path)):
        os.mkdir(log_path)

    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=1/math.e, verbose=1, patience=10, min_lr=0.0001)
    tensorboard = TensorBoard(log_dir=log_path)

    # 训练
    model.fit_generator(train_generator, steps_per_epoch=1000, epochs=8, validation_data=val_generator,
                        verbose=2, callbacks=[earlyStopping, saveBestModel, reduce_lr, tensorboard])
    # model.fit(X, y, batch_size=16, epochs=50, validation_split=0.1)
    model.save(MODEL.__name__ + '_cvd.h5')


check_path_gen()
all_train_list = abnormal_data_reject()

# tick tock start
starttime = datetime.datetime.now()
train_cat_dog(InceptionV3, (299, 299), inception_v3.preprocess_input)
endtimeV3 = datetime.datetime.now()

train_cat_dog(Xception, (299, 299), xception.preprocess_input)
endtimeXcep = datetime.datetime.now()

train_cat_dog(InceptionResNetV2, (299, 299), inception_resnet_v2.preprocess_input)
endtimeIRV2 = datetime.datetime.now()

print('InceptionV3 train time %d seconds' % (endtimeV3-starttime).seconds)
print('Xception train time %d seconds' % (endtimeXcep-endtimeV3).seconds)
print('InceptionResNetV2 train time %d seconds' % (endtimeIRV2-endtimeXcep).seconds)


