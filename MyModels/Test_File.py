import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from collections import Counter
import tensorflow as tf
'''
In this file we get the feature data from matcher of the Image_L[0] with the rest of
Image_L in the first 100 frames 
'''

def video_to_images(filename, n): # function to convert video to images
    video = cv.VideoCapture(filename)
    # print('Height of video: ', video.get(cv.CAP_PROP_FRAME_HEIGHT), '\nWidth of video: ', video.get(cv.CAP_PROP_FRAME_WIDTH), '\nFrame of video: ', video.get(cv.CAP_PROP_FRAME_COUNT))
    total_images = []
    for ii in range(n):
        # video.set(cv.CAP_PROP_POS_FRAMES, n)
        total_images.append(video.read()[1])
    video.release()
    return total_images


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 5, 5, 3])
    tf.layers.conv2d


Images_L = video_to_images("D:\QQBrowser\VideoData\\f5_dynamic_deint_L.avi", 100) # read video
Images_R = video_to_images("D:\QQBrowser\VideoData\\f5_dynamic_deint_R.avi", 100)

kernel_size = [] # blur
for ss in range(100):
    kernel_size.append((3,3)) # 8.33(2) 8.48(3) 8.32(4) 8.14(5) for kernel_size arg
Images_Ls = list(map(cv.blur, Images_L, kernel_size))
Images_Rs = list(map(cv.blur, Images_R, kernel_size))

# Images_Ls, Images_Rs = [], [] # convert to gray scale(only)
# for cot in range(100):
#     Images_Ls.append(cv.cvtColor(Images_L[cot], cv.COLOR_BGR2GRAY))
# for cot in range(100):
#     Images_Rs.append(cv.cvtColor(Images_R[cot], cv.COLOR_BGR2GRAY))

orb = cv.ORB_create()
kp_L, des_L, kp_R, des_R = [], [], [], []  # store all features
for frame_l in Images_Ls:
    kp, des = orb.detectAndCompute(frame_l, None)
    kp_L.append(kp)
    des_L.append(des)
for frame_r in Images_Rs:
    kp, des = orb.detectAndCompute(frame_r, None)
    kp_R.append(kp)
    des_R.append(des)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True) # use brute force matcher
data, labels = [], []  # train and evaluate sources
# query_ids of kp_L[0] in 100 frames (kp which shows over 20 times) :
# {55: 44, 25: 43, 31: 43, 109: 41, 75: 33, 27: 32, 41: 31, 125: 30, 97: 29, 32: 28, 7: 24, 54: 24, 15: 23, 26: 22, 65: 20}
# ->[55, 25, 31, 109, 75, 27, 41, 125, 97, 32, 7, 54, 15, 26, 65]
chosen_ids = [7, 15, 25, 26, 27, 31, 32, 41, 54, 55, 65, 75, 97, 109, 125]

for i in range(99):
    matches = bf.match(des_L[0], des_L[i + 1])  # bf.match(querydes, traindes)
    matches = sorted(matches, key=lambda x: x.distance)
    data_temp, labels_temp = [], []

    for match in matches[: 10]:
        if match.queryIdx in chosen_ids:
            data_temp.append(cv.getRectSubPix(Images_L[i + 1], (5, 5), kp_L[i + 1][match.trainIdx].pt))
            labels_temp.append(match.queryIdx)
    data.append(data_temp)
    labels.append(labels_temp)

pattern = {7:1, 15:2, 25:3, 26:4, 27:5, 31:6, 32:7, 41:8, 54:9, 55:10, 65:11, 75:12, 97:13, 109:14, 125:15}  # convert labels to [1:15]
labels2 = []
for label in labels:
    label2 = [pattern[x] if x in pattern else x for x in label]
    labels2.append(label2)
print(labels, '\n', labels2)

train_data, train_labels, eva_data, eva_labels = [], [], [], []  # training/evaluating data and label
for roi_1 in data[:80]:
    train_data.extend(roi_1)
for num_1 in labels2[:80]:
    train_labels.extend(num_1)
for roi_2 in data[80:]:
    eva_data.extend(roi_2)
for num_2 in labels2[80:]:
    eva_labels.extend(num_2)


