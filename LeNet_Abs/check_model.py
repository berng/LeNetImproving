import tensorflow as tf
from keras.datasets import mnist
import numpy as np
import keras
from keras.utils.np_utils import to_categorical
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Input, Dense, Conv2D, MaxPooling2D,AveragePooling2D,Reshape,Dropout,GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import os

import gc
import absl.logging
import tracemalloc

absl.logging.set_verbosity(absl.logging.ERROR)
gc.enable()
# tracemalloc.start()
# +
# Custom callback

import matplotlib.pyplot as pp
import numpy as np

img_rows=28
img_cols=28
batch_size=128
epochs=100

model = tf.keras.models.load_model('model')
model.summary()
model.compile(loss=tf.keras.losses.categorical_crossentropy,
          optimizer='adam',
          metrics=['accuracy'])



import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
import tensorflow.keras as keras
import os
import numpy as np

(x_all,y_all),(x_test,y_test) = datasets.mnist.load_data()
x_all = tf.pad(x_all, [[0, 0], [2,2], [2,2]])
x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]])
print('shape:',x_all.shape)
x_valid = x_all[x_all.shape[0]*80//100:,:,:]
x_train = x_all[:x_all.shape[0]*80//100,:,:]
y_valid = y_all[y_all.shape[0]*80//100:]
y_train = y_all[:y_all.shape[0]*80//100]

# for conv2D expand dims - add color level
x_train = tf.expand_dims(x_train, axis=3, name=None)
x_valid = tf.expand_dims(x_valid, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)
print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)

y_train=tf.keras.utils.to_categorical(y_train,num_classes=10)
y_test=tf.keras.utils.to_categorical(y_test,num_classes=10)
y_valid=tf.keras.utils.to_categorical(y_valid,num_classes=10)

print('train:',model.evaluate(x_train, y_train,verbose=0))
print('val:',model.evaluate(x_valid, y_valid,verbose=0))
res=model.evaluate(x_test, y_test,verbose=0)
print('test:',res)
tot=y_test.shape[0]
acc=int(tot*res[1])
from statsmodels.stats.proportion import proportion_confint
wilson_interval = proportion_confint(acc, tot, method = 'wilson')
print('wilson interval:')
print(wilson_interval)
