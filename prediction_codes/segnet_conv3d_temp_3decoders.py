import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras
import cv2
import numpy as np
from keras.models import *
from keras.layers import *
import image_utils
import scipy.io as sio

def predicted_output(model,X_dev, frames, height=68,width=68,n_ops=2):
    pr = model.predict(X_dev)
    print(np.array(pr).shape)
    pr = np.argmax(pr,axis=-1)
    pred = np.zeros((pr.shape[0],pr.shape[1], frames, height,width),dtype = int)
    print(np.array(pred).shape)
    for i in range(n_ops-1):
      pred[:,:,:,:,:] = ((pr==i+1)*255).astype(int)
    return pred

height = 68
width = 68
n_ops = 2
segnet_model = load_model('./model.weights')
save_path=('./')
#sub = ['F1','F2','F3','F4','F5','M1','M2','M3','M4','M5']
vidcap = cv2.VideoCapture('./F2_400.avi')
a = []
success = True
while success:
	success,image = vidcap.read()
	a.append(image)
a = np.array(a)
b = np.zeros((a.shape[0]-1,68,68,3))
for i in range(a.shape[0]-1):
	b[i,:,:,:] = a[i]
b=b.astype('float64')
b=b/255
frames = b.shape[0]
preds = predicted_output(segnet_model,b,frames,height,width,n_ops=2)
sio.savemat(save_path +'F2_400.mat', {'pred':preds,'name':names_test})

