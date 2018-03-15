# -*- coding: utf-8 -*-

from keras.models import *
import pandas as pd
from keras.preprocessing.image import *
from keras.applications import *
from tqdm import tqdm
import cv2


def predict_cat_dog(model_name, img_size, preprocess_func):
    batch_size = 100

    width = img_size[0]
    height = img_size[1]
    
    model = load_model(model_name)

    gen = ImageDataGenerator(preprocessing_function=preprocess_func)
    test_generator = gen.flow_from_directory("./Dataset/test2", img_size, shuffle=False,
                                             batch_size=batch_size, class_mode=None)

    y_test = model.predict_generator(test_generator, test_generator.samples // batch_size)
    y_test = y_test.clip(min=0.005, max=0.995)

    # 产生submission
    df = pd.read_csv("sample_submission.csv")

    for i, fname in enumerate(test_generator.filenames):
        index = int(fname[fname.rfind('/') + 1:fname.rfind('.')])
        df.set_value(index - 1, 'label', y_test[i])

    df.to_csv(model_name.split('_')[0]+'_pred.csv', index=None)
    # df.head(10)


predict_cat_dog('InceptionV3_cvd.h5', (299, 299), inception_v3.preprocess_input)
predict_cat_dog('Xception_cvd.h5', (299, 299), xception.preprocess_input)
predict_cat_dog('InceptionResNetV2_cvd.h5', (299, 299), inception_resnet_v2.preprocess_input)






