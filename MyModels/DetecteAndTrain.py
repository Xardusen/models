import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def video_to_images(filename, n):
    video = cv.VideoCapture(filename)
    print('Height of video: ', video.get(cv.CAP_PROP_FRAME_HEIGHT), '\nWidth of video: ', video.get(cv.CAP_PROP_FRAME_WIDTH), '\nFrame of video: ', video.get(cv.CAP_PROP_FRAME_COUNT))
    total_images = []
    for i in range(n):
        # video.set(cv.CAP_PROP_POS_FRAMES, n)
        total_images.append(video.read()[1])
    video.release()
    return total_images


# read video
Images_L = video_to_images("D:\QQBrowser\VideoData\\f5_dynamic_deint_L.avi", 100)
Images_R = video_to_images("D:\QQBrowser\VideoData\\f5_dynamic_deint_R.avi", 100)

# extract key points and descriptors
orb = cv.ORB_create()
kp_L, des_L, kp_R, des_R = [], [], [], []  # store all features
for frame_l in Images_L:
    kp, des = orb.detectAndCompute(frame_l, None)
    kp_L.append(kp)
    des_L.append(des)
for frame_r in Images_R:
    kp, des = orb.detectAndCompute(frame_r, None)
    kp_R.append(kp)
    des_R.append(des)

# match the descriptors
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_L[0], des_R[0])
print('Total matches : ', len(matches))
matches = sorted(matches, key=lambda x: x.distance)
img_match = cv.drawMatches(Images_L[0], kp_L[0], Images_R[0], kp_R[0], matches[:50], None, flags=2)
img_match = cv.cvtColor(img_match, cv.COLOR_BGR2RGB)
# plt.imshow(img_match), plt.show()
print(len(des_L[0][0]))
print(kp_L[0][0].pt[0], kp_L[0][0].pt[1])
