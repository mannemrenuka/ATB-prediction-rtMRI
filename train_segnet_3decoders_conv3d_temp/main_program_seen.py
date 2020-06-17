import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session  
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
set_session(tf.Session(config=config)) 
from keras.models import *
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
import glob
#import itertools
import numpy as np
import sys
from keras import optimizers
import model 
import image_utils
import scipy.io as sio


subs = ['F1','F2','F3','F4','F5','M1','M2','M3','M4','M5']
path = 'segnet_conv3d_temp_seen_3decoders'
model_dir = './'+path+'/models/'
in_dir = './'+path+'/110_videos/' ## videos directory
out_dir = './'+path+'/mat_110/' ## masks matfiles directory

n_ops = 2
frames = 24
height=68
width=68
epoch =30
new = 1
fixed_dim = 68

if not os.path.exists(model_dir):
    print("creating model directory ")
    os.makedirs(model_dir)
actual_set=np.array([342,391,392,393,394,395,397,398,399,406,413])
train_matrix = actual_set[0:9]
val_matrix = actual_set[9:10]
test_matrix = actual_set[10:11]

X_train,Y_train,X_val,Y_val, X_test,Y_test,name,name_val,name_test=image_utils.imageSegmentationGenerator(in_dir,out_dir,n_ops,frames,height,width,train_matrix,val_matrix,test_matrix)

print (X_train.shape)
print (Y_train[0].shape)
print (Y_train[1].shape)
print (Y_train[2].shape)
print (X_val.shape)
print (Y_val[0].shape)
print (Y_val[1].shape)
print (Y_val[2].shape)
print (X_test.shape)
print (Y_test[0].shape)
print (Y_test[1].shape)
print (Y_test[2].shape)
print (type(Y_test[2]))
print (name.shape)
print (name_val.shape)
print (name_test.shape)
	
print('*********new model fitting*********')
segnet_model=model.Segnet(n_ops ,height, width, frames,fixed_dim)
#segnet_model = load_model(model_dir+'model_'+test_matrix+'.weights')
segnet_model.summary()
#for layer in segnet_model.layers:
#    print(layer.output_shape)
										
segnet_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#target1 = tf.placeholder(dtype='float',shape=(None, 24, 68, 68, 2))
#target2 = tf.placeholder(dtype='float',shape=(None, 24, 68, 68, 2))
#target3 = tf.placeholder(dtype='float',shape=(None, 24, 68, 68, 2))
#segnet_model.compile(loss=model.cust_mse,optimizer='adam',metrics=['accuracy'],target_tensor=[target1,target2,target3])
#segnet_model.compile(loss=model.cust_mse,optimizer='adam',metrics=['accuracy'],target_tensors=[target1,target2,target3])
callbacks = [ModelCheckpoint(model_dir+'model_413.weights', monitor='val_loss', verbose=0, save_best_only=True, mode = 'auto',period=1)]		
#callbacks = [EarlyStopping(monitor='val_loss', patience=10, min_delta = 0.00001),ModelCheckpoint(model_dir+'model_'+test_matrix+'.weights', monitor='val_loss', verbose=0, save_best_only=True, mode='auto', period=1)]
history=segnet_model.fit(X_train, Y_train, epochs=epoch,batch_size=4,validation_data = (X_val, Y_val),verbose=1,callbacks=callbacks)
#segnet_model.save(model_dir+'/model_'+str(count)+'.weights')
#preds = image_utils.predicted_output(segnet_model,X_train,frames,height,width,n_ops)
#sio.savemat(model_dir+'train_pred_'+test_matrix+'.mat', {'pred':preds,'name':name})
preds = image_utils.predicted_output(segnet_model,X_test,frames,height,width,n_ops)
sio.savemat(model_dir+'test_pred_413.mat', {'pred':preds,'name':name_test})


