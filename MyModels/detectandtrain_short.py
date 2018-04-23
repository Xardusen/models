import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def video_to_images(filename, n): # function to convert video to images
    video = cv.VideoCapture(filename)
    # print('Height of video: ', video.get(cv.CAP_PROP_FRAME_HEIGHT), '\nWidth of video: ', video.get(cv.CAP_PROP_FRAME_WIDTH), '\nFrame of video: ', video.get(cv.CAP_PROP_FRAME_COUNT))
    total_images = []
    for i in range(n):
        # video.set(cv.CAP_PROP_POS_FRAMES, n)
        total_images.append(video.read()[1])
    video.release()
    return total_images

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
interMatch, same_match = [], []  # same_match starts from the third frame

for i in range(98):
    matches_1 = bf.match(des_L[i], des_L[i+ 1])  # point location1
    # print('Total matches I : ', len(matches_1))
    matches_1 = sorted(matches_1, key=lambda x: x.distance)
    # fea_img_1 = []
    # for y in range(10):
    #     fea_img_1.append(matches_1[y].trainIdx)
    # print(fea_img_1)
    # pic1 = cv.drawMatches(Images_L[i], kp_L[i], Images_L[i + 1], kp_L[i + 1], matches_1[: 10], None, flags=2)
    # cv.imshow('ma_1', pic1)

    matches_2 = bf.match(des_L[i + 1], des_L[i + 2])  # point location2
    # print('Total matches II : ', len(matches_2))
    matches_2 = sorted(matches_2, key=lambda x: x.distance)
    # fea_img_2 = []
    # for yy in range(10):
    #     fea_img_2.append(matches_2[yy].queryIdx)
    # print(fea_img_2)
    # pic2 = cv.drawMatches(Images_L[i + 1], kp_L[i + 1], Images_L[i + 2], kp_L[i + 2], matches_2[: 10], None, flags=2)
    # cv.imshow('ma_2', pic2)
    interMatch_tmp, same_match_temp = [], []
    for ma in matches_1[:10]:
        for mb in matches_2:
            if ma.trainIdx == mb.queryIdx:
                interMatch_tmp.append(mb)
    interMatch.append(len(interMatch_tmp))

    same_match_temp = list(set(interMatch_tmp).intersection(set(matches_2[:10])))
    same_match.append(len(same_match_temp))


    # for item in interMatch:
    #     print(item.queryIdx, end=',')
    # print('\n')
    # for item_2 in matches_1[:10]:
    #     print(item_2.trainIdx, end=',')
    # ma1 = cv.drawMatches(Images_L[55], kp_L[55], Images_L[56], kp_L[56], matches_1[:10], None, flags=2)
    # ma2 = cv.drawMatches(Images_L[56], kp_L[56], Images_L[57], kp_L[57], interMatch, None, flags=2)
    # cv.imshow('1', ma1)
    # cv.imshow('2', ma2)


print(interMatch, '\naverage : ', sum(interMatch)/98, '\n', same_match, '\naverage : ', sum(same_match)/98,)

cv.waitKey(0)

# intersection = list(set(fea_img_1).intersection(set(fea_img_2)))  # intersection
# print(intersection)
