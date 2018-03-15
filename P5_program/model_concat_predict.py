# -*- coding: utf-8 -*-
import h5py
import numpy as np
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *

import pandas as pd
from keras.preprocessing.image import *

# bottleneck产生测试特征
np.random.seed(2017)

X_train = []
X_test = []
y_pred = []

for filename in ["gap_InceptionV3.h5", "gap_Xception.h5", "gap_InceptionResNetV2.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)

# 获取保存的模型
model = load_model('model_concat_cvd.h5')

y_pred = model.predict(X_test, verbose=1)
y_pred = y_pred.clip(min=0.005, max=0.995)


# 产生submission
df = pd.read_csv("sample_submission.csv")

gen = ImageDataGenerator()
test_generator = gen.flow_from_directory("/home/autel/Dataset/Cat_vs_Dog/test2", (224, 224), shuffle=False,
                                         batch_size=16, class_mode=None)

for i, fname in enumerate(test_generator.filenames):
    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
    df.set_value(index-1, 'label', y_pred[i])

df.to_csv('model_concat_pred.csv', index=None)
df.head(10)