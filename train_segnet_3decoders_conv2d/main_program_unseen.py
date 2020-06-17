import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
import matplotlib.pyplot as plt
import sys
from matplotlib.pyplot import imread
from keras import optimizers
import model 
import image_utils
import scipy.io as sio



subs = ['F1','F2','F3','F4','F5','M1','M2','M3','M4','M5']
path = 'segnet_conv2d_unseen_3decoders_binary'
model_dir = './'+path+'/models/'
in_dir = './'+path+'/110_videos/' ## videos directory
out_dir = './'+path+'/mat_110/' ## masks matfiles directory

n_ops = 2
height=68
width=68
epoch =30
new = 1
segnet_model=model.Segnet(n_ops ,height, width)
segnet_model.summary()	

#segnet_model=model.Segnet(n_ops ,height, width)
#for layer in segnet_model.layers:
    #print(layer.output_shape)
if not os.path.exists(model_dir):
    print("creating model directory ")
    os.makedirs(model_dir)
subs = ['F1','F2','F3','F4','F5','M1','M2','M3','M4','M5']
model_dir = '/home/user/home2/data/navaneetha/models/segnet_conv3d_fixed_unseen_3decoders_binary/'
in_dir = '/home/user/home2/data/navaneetha/110_videos/' ## videos directory
out_dir = '/home/user/home2/data/navaneetha/mat_110/' ## masks matfiles directory
mat_dir = '/home/user/home2/data/navaneetha/'
if not os.path.exists(model_dir):
    print("creating model directory ")
    os.makedirs(model_dir)
actual_set=np.array([342,391,392,393,394,395,397,398,399,406,413])
for i in range(0,len(subs)):
	train_matrix = [subs[j] for j in list(range(0,i))+list(range(i+1,len(subs)))]
	print(train_matrix)
	test_matrix = subs[i]
	print(test_matrix)
	X_train,Y_train,X_test,Y_test,name,name_test=image_utils.imageSegmentationGenerator(in_dir , out_dir , n_ops , height , width , train_matrix,test_matrix)
	print ('X_train.shape',X_train.shape)
	print ('Y_train[0].shape',Y_train[0].shape)
	print (Y_train[1].shape)
	print (Y_train[2].shape)
	print (X_test.shape)
	print (Y_test[0].shape)
	print (Y_test[1].shape)
	print (Y_test[2].shape)
	print (type(Y_test[2]))
	print (name.shape)
	#print (name_val.shape)
	print (name_test.shape)		
	print('*********new model fitting*********')
	#######segnet_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])#,target_tensors= [target1,target2,target3])
	#segnet_model.compile(loss=model.cust_mse,optimizer='adam')
	##segnet_model.compile(optimizer='adam')
	print('compilation completed')
	######callbacks = [ModelCheckpoint(model_dir+'model_'+test_matrix+'.weights', monitor='val_loss', verbose=0, save_best_only=True, mode = 'auto',period=1)]		
	print('callbacks completed')
	######history=segnet_model.fit(X_train, (Y_train), epochs=epoch,batch_size=4,validation_split = 0.1,verbose=1,callbacks=callbacks)


