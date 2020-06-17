import keras
from keras.models import *
from keras.layers import*
import cv2
import glob
import os
import sys
import itertools
from keras import optimizers
from keras import models


'''from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import convolution2D,MaxPooling2D,UpSampling2D, Conv2DTranspose, Conv2D
from keras.layers.normalization import BatchNormalization'''


import json
import numpy as np
import time
from keras import backend as K


def Segnet( n_labels=2 ,img_h=68 , img_w=68, frames = 24, fixed_dim =68):
	kernel = 3
	ker_fr = 2
	img_in =  keras.Input(shape=( frames, img_h, img_w, 3))
	c1 = Conv3D(64, (ker_fr, kernel,kernel), padding='same')(img_in)
	b1 = BatchNormalization()(c1)
	a1 = Activation('relu')(b1)
	c2 = Conv3D(64, (ker_fr,kernel,kernel), padding='same')(a1)
	b2 = BatchNormalization()(c2)
	a2 = Activation('relu')(b2)
	m1 = MaxPooling3D()(a2)
	# 12*34*34 
	c3 = Conv3D(128, (ker_fr, kernel, kernel), padding='same')(m1)
	b3 = BatchNormalization()(c3)
	a3 = Activation('relu')(b3)
	c4 = Conv3D(128, (ker_fr, kernel, kernel), padding='same')(a3)
	b4 = BatchNormalization()(c4)
	a4 = Activation('relu')(b4)
	m2 = MaxPooling3D()(a4)
	# 6*17*17
	c5 = Conv3D(256, (ker_fr, kernel, kernel), padding='same')(m2)
	b5 = BatchNormalization()(c5)
	a5 = Activation('relu')(b5)
	c6 = Conv3D(256, (ker_fr, kernel, kernel), padding='same')(a5)
	b6 = BatchNormalization()(c6)
	a6 = Activation('relu')(b6)
	c7 = Conv3D(256, (ker_fr, kernel, kernel), padding='same')(a6)
	b7 = BatchNormalization()(c7)
	a7 = Activation('relu')(b7)
	m3 = MaxPooling3D()(a7)
	# 3*8*8
	
	####decoder
	###for mask1
	#6x17x17
	c8 = Conv3DTranspose( 512 , kernel_size=(ker_fr,kernel,kernel),strides=(2,2,2))(m3)
	c9 = Conv3D(512, (ker_fr, kernel, kernel), padding='same')(c8)
	b8 = BatchNormalization()(c9)
	a8 = Activation('relu')(b8)
	# 12*34*34	
	u1 = UpSampling3D()(a8)
	c12 = Conv3D(256, (ker_fr, kernel, kernel), padding='same')(u1)
	b11 = BatchNormalization()(c12)
	a11 = Activation('relu')(b11)
	# 24*68*68
	u2 = UpSampling3D()(a11) ## bs*24*68*68*
	c15 = Conv3D(128, (ker_fr, kernel, kernel), padding='same')(u2)
	b14 = BatchNormalization()(c15)
	a14 = Activation('relu')(b14)
	c18 = Conv3D(n_labels, (1, 1, 1), padding='valid')(a14)
	b16 = BatchNormalization()(c18)
	s1 = Activation('softmax')(b16)	
	###for mask2
        #6x17x17
	mc1 = Conv3DTranspose( 512 , kernel_size=(ker_fr,kernel,kernel) ,  strides=(2,2,2))(m3)
	mc2 = Conv3D(512, (ker_fr, kernel, kernel), padding='same')(mc1)
	mb1 = BatchNormalization()(mc2)
	ma1 = Activation('relu')(mb1)	
	# 12*34*34	
	mu1 = UpSampling3D()(ma1)
	mc3 = Conv3D(256, (ker_fr, kernel, kernel), padding='same')(mu1)
	mb2 = BatchNormalization()(mc3)
	ma2 = Activation('relu')(mb2)
	# 24*68*68
	mu2 = UpSampling3D()(ma2) ## bs*24*68*68*
	mc4 = Conv3D(128, (ker_fr, kernel, kernel), padding='same')(mu2)
	mb3 = BatchNormalization()(mc4)
	ma3 = Activation('relu')(mb3)
	mc5 = Conv3D(n_labels, (1, 1, 1), padding='valid')(ma3)
	mb4 = BatchNormalization()(mc5)
	s2 = Activation('softmax')(mb4)	
	###for mask3
        #6x17x17
	mc6 = Conv3DTranspose( 512 , kernel_size=(ker_fr,kernel,kernel) ,  strides=(2,2,2))(m3)
	mc7 = Conv3D(512, (ker_fr, kernel, kernel), padding='same')(mc6)
	mb5 = BatchNormalization()(mc7)
	ma5 = Activation('relu')(mb5)	
	# 12*34*34	
	mu3 = UpSampling3D()(ma5)
	mc8 = Conv3D(256, (ker_fr, kernel, kernel), padding='same')(mu3)
	mb6 = BatchNormalization()(mc8)
	ma6 = Activation('relu')(mb6)
	# 24*68*68
	mu4 = UpSampling3D()(ma6) ## bs*24*68*68*
	mc9 = Conv3D(128, (ker_fr, kernel, kernel), padding='same')(mu4)
	mb7 = BatchNormalization()(mc9)
	ma7 = Activation('relu')(mb7)
	mc10 = Conv3D(n_labels, (1, 1, 1), padding='valid')(ma7)
	mb8 = BatchNormalization()(mc10)
	s3 = Activation('softmax')(mb8)	
	model = keras.models.Model(img_in,[s1,s2,s3])
	return model
	
def cust_mse(atrue,apred):
	l1 = K.mean(K.square(atrue[0]-apred[0]))
	l2 = K.mean(K.square(atrue[1]-apred[1]))
	l3 = K.mean(K.square(atrue[2]-apred[2]))
	return l1+l2+l3

