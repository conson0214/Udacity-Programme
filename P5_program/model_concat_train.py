# -*- coding: utf-8 -*-
import h5py
import numpy as np
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
from keras.callbacks import *
import datetime

# bottleneck产生训练特征
# np.random.seed(2017)

X_train = []
X_test = []

starttime = datetime.datetime.now()

for filename in ["gap_InceptionV3.h5", "gap_Xception.h5", "gap_InceptionResNetV2.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)

X_train, y_train = shuffle(X_train, y_train)

input_tensor = Input(X_train.shape[1:])
x = Dropout(0.5)(input_tensor)
x = Dense(1, activation='sigmoid')(x)
model = Model(input_tensor, x)

model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

log_path = './model_concat_tensor_log'
tensorboard = TensorBoard(log_dir=log_path)


model.fit(X_train, y_train, batch_size=128, nb_epoch=8, validation_split=0.2, callbacks=[tensorboard])
endtime = datetime.datetime.now()
print("model concat train time %d seconds" % (endtime-starttime).seconds)

# 模型保存
# json_string = model.to_json()
# open('model_concat_cvd_architecture.json', 'w').write(json_string)
# model.save_weights('model_concat_cvd_weights.h5')
model.save('model_concat_cvd.h5')

# from keras.utils import plot_model
# from keras.utils.vis_utils import model_to_dot
# model_to_dot(model, show_shapes=True).write('./test.dot')

