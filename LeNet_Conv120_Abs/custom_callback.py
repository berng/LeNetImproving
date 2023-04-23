import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
import tensorflow.keras as keras
import os
import numpy as np
f=open('./log_params.dat','wt')
f.close()

(x_all,y_all),(x_test,y_test) = datasets.mnist.load_data()
x_all = tf.pad(x_all, [[0, 0], [2,2], [2,2]])
x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]])

x_valid = x_all[x_all.shape[0]*80//100:,:,:]
x_train = x_all[:x_all.shape[0]*80//100,:,:]
y_valid = y_all[y_all.shape[0]*80//100:]
y_train = y_all[:y_all.shape[0]*80//100]

# for conv2D expand dims - add color level
x_train = tf.expand_dims(x_train, axis=3, name=None)
x_valid = tf.expand_dims(x_valid, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)

LR=0.1
class CustomCallback(keras.callbacks.Callback):
    def __init__(self, patience=5, name='model'):
        super(CustomCallback, self).__init__()
        self.patience=patience
        self.max_expected_acc=-1e10
        self.save_model_name=name
        self.best_epoch=-1
        
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        self.v_max_found=False
    def on_epoch_begin(self, epoch, logs=None):
        print("ep:",epoch+1,flush=True)
    def on_batch_begin(self, batch, logs=None):
        if batch%4==0: print("\r-",end='',flush=True)
        if batch%4==1: print("\r\\",end='',flush=True)
        if batch%4==2: print("\r|",end='',flush=True)
        if batch%4==3: print("\r/",end='',flush=True)

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        res1=self.model.evaluate(x=x_valid[:y_valid.shape[0]//2],y=y_valid[:y_valid.shape[0]//2],verbose=0)
        res2=self.model.evaluate(x=x_valid[y_valid.shape[0]//2:],y=y_valid[y_valid.shape[0]//2:],verbose=0)

        self.y2pred=self.model.predict(x_valid,verbose=0)
        self.bs_bound,self.m_bs_bound,self.s_bs_bound=self.get_lower_bound(self.y2pred,y_valid)

        self.full_acc=np.minimum(res1[1],res2[1])
        self.val_acc=logs['val_accuracy']
        
        self.expected_acc=np.minimum(res1[1],res2[1])-self.s_bs_bound #correct min() for sigma
        f=open('./log_params.dat','at')
        f.write('lr:'+str(LR)+'\t')
        f.write('e:'+str(epoch+1)+'\t')
        f.write('ta:'+str(self.full_acc)+'\t')
        f.write('va:'+str(self.val_acc)+'\t')
        f.write('ea:'+str(self.expected_acc)+'\t')
        f.write('logs:'+str(logs)+'\n')
        f.close()

        if self.expected_acc>self.max_expected_acc:
          self.best_epoch=epoch
          print('at epoch ',epoch,'\n we have acc at train: ',self.full_acc, 
                ' val: ',self.val_acc,
                ' and expect ',self.expected_acc,' at test')
          self.max_expected_acc=self.expected_acc
          print('save as optimal model')
#          tf.keras.models.save_model(self.model, './optimal/'+self.save_model_name)
          tf.keras.models.save_model(self.model, './'+self.save_model_name)
          f=open('./'+self.save_model_name+'/params.dat','wt')
          f.write('learning rate:'+str(LR)+'\n')
          f.write('best epoch:'+str(epoch+1)+'\n')
          f.write('train acc:'+str(self.full_acc)+'\n')
          f.write('val acc:'+str(self.val_acc)+'\n')
          f.write('expected acc:'+str(self.expected_acc)+'\n')
          f.close()
        if epoch>self.best_epoch+10:
          self.model.stop_training=True

#bootstraped distribution of errors
    def get_lower_bound(self,pred_cat,src_cat):
      accs=[]
      pred=np.argmax(pred_cat,axis=1)
      src=src_cat
      for i in range(100000):
       idx=np.random.randint(0,pred.shape[0],pred.shape[0])
       pt=pred[idx]
       st=src[idx]
       acc=0
       acc=(pt[pt==st].shape[0]/pt.shape[0])
       accs.append(acc)
      accs=np.array(accs)
      cur_sigma=1
      return accs.mean()-cur_sigma*accs.std(),accs.mean(),accs.std()  # alpha=0.0001


