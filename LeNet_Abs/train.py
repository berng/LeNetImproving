# https://medium.com/analytics-vidhya/lenet-with-tensorflow-a35da0d503df
# https://www.kaggle.com/code/blurredmachine/lenet-architecture-a-complete-guide

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
import tensorflow.keras as keras
import os
import gc
import numpy as np

import custom_callback as cc
from lenet import lenet

# +
def train(model_func,src,y,src_v,y_v,model_name='model',saveOpt=False,epochs=10,custom_callback=''):
    model=model_func
    ycat=y #tf.keras.utils.to_categorical(y,num_classes=10)
    model.summary()
    if saveOpt:
        history=model.fit(src,ycat,epochs=epochs,
                      validation_data=[src_v,y_v],
                      verbose=0,
                      callbacks=[custom_callback])
        model=tf.keras.models.load_model('./'+model_name)        
    else:
        history=model.fit(src,ycat,epochs=epochs,
                      validation_data=[src_v,y_v],
                      verbose=0)
        os.system('mkdir optimal')
        tf.keras.models.save_model(model, './'+model_name)    
    return model,history


# LeNet
# tf.keras.utils.plot_model(model_lenet,show_layer_activations=True,show_shapes=True,to_file='./model.png')
import matplotlib.pyplot as pp
import numpy as np
def show_mod_hist(model_list,history_list,legend_list,N=3):
    fig,axs=pp.subplots(3,N,figsize=(3*N,7))
    for i in range(1):
        axs[0].set_ylim(0,1)
        axs[1].set_ylim(0,0.2)
        axs[2].set_ylim(0.95,1)
        axs[2].set_xlabel('epoch')
        axs[0].set_ylabel('loss')
        axs[1].set_ylabel('loss')
        axs[2].set_ylabel('accuracy')
        
        axs[0].plot(history_list[i].history['loss'],label='train')
        axs[0].plot(history_list[i].history['val_loss'],label='validation')
        axs[0].legend()

        axs[0,].set_title(legend_list[i])
        
        axs[1].plot(history_list[i].history['loss'])
        axs[1].plot(history_list[i].history['val_loss'])

        axs[2].plot(history_list[i].history['accuracy'])
        axs[2].plot(history_list[i].history['val_accuracy'])
        
# Epochs=100
#for cc.LR in [0.001,0.003,0.0001,0.0003,0.00001,0.00003,0.000001,0.000003,0.0000001]:
model=lenet(cc.x_train.shape[1:],lr=0.01)
prev_lr=0
#for cc.LR in [0.01,0.001,0.0001,0.00001,0.000001]:
for cc.LR in [0.001,0.0001,0.00001,0.000001]:
 print("train learing rate",cc.LR)
 if (prev_lr>0):
  model=tf.keras.models.load_model('model')
 model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cc.LR), 
                loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
 my_ccb=cc.CustomCallback(patience=1,name='model')
 model_lenet,history_lenet = train(model,cc.x_train,cc.y_train,cc.x_valid,cc.y_valid,
                                  model_name='model',
                                  epochs=1000, #stop after not finding best results during 50 epochs, see callback
                                  saveOpt=True,
                                  custom_callback=my_ccb)
 prev_lr=cc.LR
 show_mod_hist([model_lenet],
              [history_lenet],
              ['Lenet+Abs']
              ,N=1)
 pp.savefig('train-lr'+str(cc.LR)+'.png')
 gc.collect()

print('LeNet+Abs: loss,accuracy:',model_lenet.evaluate(cc.x_test,cc.y_test,verbose=0),
                              model_lenet.evaluate(cc.x_train,cc.y_train,verbose=0),
                              model_lenet.evaluate(cc.x_valid,cc.y_valid,verbose=0))
