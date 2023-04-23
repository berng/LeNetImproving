# https://medium.com/analytics-vidhya/lenet-with-tensorflow-a35da0d503df
# https://www.kaggle.com/code/blurredmachine/lenet-architecture-a-complete-guide

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
import tensorflow.keras as keras
import os
import numpy as np

def lenet(x_train_shape,deep=0,lr=1e-3):
  input = layers.Input(shape=x_train_shape)
  data = input
  data = layers.Conv2D(6, 5, padding='same', activation='linear')(data)
  data = tf.math.abs(data)
  data = layers.AveragePooling2D(2)(data)
  data = tf.math.abs(data)
  data = layers.Conv2D(64, 5, activation='linear')(data)
  data = tf.math.abs(data)
  data = layers.AveragePooling2D(2)(data)
  data = tf.math.abs(data)
  data = layers.Conv2D(120, 5, activation='linear')(data)
  data = tf.math.abs(data)
  data = layers.AveragePooling2D(2)(data)
  data = tf.math.abs(data)
  data = layers.Conv2D(120,1, activation='linear')(data)
  data = tf.math.abs(data)
  data = layers.Flatten()(data)
  for i in range(deep):
    data = layers.Dense(84, activation='linear')(data)
    data = tf.math.abs(data)
  data = layers.Dense(84, activation='linear')(data)
  data = tf.math.abs(data)
  data_out = layers.Dense(10, activation='softmax')(data)
  model = tf.keras.models.Model(inputs=input,outputs=data_out)
  return model
