# ATB-prediction-rtMRI
The repository contains the traning and evaluation codes for air-tissue boundary (ATB) segmentation in real time magnetic resonance imaging (rtMRI) video.\
At first, SegNet is used to semantically segment the rtMRI image into 3 masks which correspond to the 3 ATBs. \
Then, the 3 masks are used to obtain the ATBs using contour prediction approach. \
We use two different types of SegNet architectures: \
1) SegNet with 3 reduced decoders with 2D convolutional layers. \
2) SegNet with 3 reduced decoders with 3D convolutional layers which uses temporal information of the rtMRI video. \
Discription about the folders: \
train_segnet_3decoders_conv2d: Using SegNet with 2D CNN. \
train_segnet_3decoders_conv3d_temp: Using SegNet with 3D CNN. \
prediction codes: To predict masks for a given video F2_400.avi. \ 
The required trained models can be downloaded in below google drive links: \
SegNet with 2D CNN: https://drive.google.com/file/d/1SBFOm2ULVJW7cmdpeC-85fl3bWU4wyio/view?usp=sharing \
SegNet with 3D CNN: https://drive.google.com/file/d/1x6EW6AEMRsw-ZOMsYsYgNPtpN7ZAKVq_/view?usp=sharing \
Description about the codes: \
main_program_seen/unseen.py : To train the model with the given input rtMRI images and output binary masks in seen and unseen subject conditions. \
image_utils.py : to divide dataset into train, validation and test set. \
model.py : Provides the architecure of the model. \
Seen subejct condition: \
    Total 110 videos from 10 subjects \
    Training : 9 videos x 10 subjects = 90 videos \
        9 videos : 342,391,392,393,394,395,397,398,399 \
    Validation : 1 video x 10 subjects = 10 videos \
        Video = 406 all subjects \
    Testing :  1 video x 10subjects = 10 videos \
        Video = 413 all subjects \
Unseen subejct condition: \
    Leave one subejct out experimental setup - 10 fold cross validation setup as there are 10 subjects. \
    Training : 9 subjects x 11 videos  = 99 videos \
    Validation : 0.1 percent of training data is considered to be validation data \
    Testing :  1 unseen subject x 11videos = 11 videos \


