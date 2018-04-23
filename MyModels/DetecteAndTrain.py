import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def video_to_images(filename, n):
    video = cv.VideoCapture(filename)
    # print('Height of video: ', video.get(cv.CAP_PROP_FRAME_HEIGHT), '\nWidth of video: ', video.get(cv.CAP_PROP_FRAME_WIDTH), '\nFrame of video: ', video.get(cv.CAP_PROP_FRAME_COUNT))
    total_images = []
    for i in range(n):
        # video.set(cv.CAP_PROP_POS_FRAMES, n)
        total_images.append(video.read()[1])
    video.release()
    return total_images
def kp_to_feature_block(key_points_1, key_points_2, des_1, des_2):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    block_number = 50

    # set the coordinates, blocks and labels for features
    kp_cds_1, kp_cds_2, kp_bls_1, kp_bls_2, kp_labs_1, kp_labs_2 = [], [], [] ,[], [], []
    matches = sorted(bf.match(des_1[0], des_1[1]), key=lambda x: x.distance)

    match_processing = []
    for n in range(50):
        match_processing.append(key_points_1[matches[:50][n].queryIdx].pt)
    kp_cds_1.append(match_processing)  # get kp_cds_1[0]
    bls_processing = []
    for coord in kp_cds_1[0]:
        bls_processing.append(cv.getRectSubPix(Images_L[0], (5, 5), coord))
    kp_bls_1.append(bls_processing)  # get kp_bls_1[0]



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
matches = bf.match(des_L[50], des_R[50])
print('Total matches : ', len(matches))
matches = sorted(matches, key=lambda x: x.distance)
img_match = cv.drawMatches(Images_L[50], kp_L[50], Images_R[50], kp_R[50], matches[:50], None, flags=2)
img_match = cv.cvtColor(img_match, cv.COLOR_BGR2RGB)
# plt.imshow(img_match), plt.show()



cv.waitKey(0)
