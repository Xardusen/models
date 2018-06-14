import numpy as np
import cv2 as cv
from collections import Counter
import math


def video_to_images(filename, n): # function to convert video to images
    video = cv.VideoCapture(filename)
    # print('Height of video: ', video.get(cv.CAP_PROP_FRAME_HEIGHT), '\nWidth of video: ', video.get(cv.CAP_PROP_FRAME_WIDTH), '\nFrame of video: ', video.get(cv.CAP_PROP_FRAME_COUNT))
    total_images = []
    for ii in range(n):
        # video.set(cv.CAP_PROP_POS_FRAMES, n)
        total_images.append(video.read()[1])
    video.release()
    return total_images


def main():
    Images_L = video_to_images("D:\QQBrowser\VideoData\\f5_dynamic_deint_L.avi", 1100) # read video
    Images_R = video_to_images("D:\QQBrowser\VideoData\\f5_dynamic_deint_R.avi", 1100)

    kernel_size = [] # blur
    for image in Images_L:
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
    for frame_L in Images_Ls:
        kp, des = orb.detectAndCompute(frame_L, None)
        kp_L.append(kp)
        des_L.append(des)
    for frame_R in Images_Rs:
        kp, des = orb.detectAndCompute(frame_R, None)
        kp_R.append(kp)
        des_R.append(des)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True) # use brute force matcher
    data, labels = [], []  # train and evaluate sources

    featureIds_1, featureIds_2 = [], []  # calculate the chosen_ids
    for a in range(1, 1000):
        matches = bf.match(des_L[0], des_L[a])
        matches = sorted(matches, key=lambda  x: x.distance)
        for item in matches[: 20]:
            featureIds_1.append(item.queryIdx)
    for b in range(11, 1011):
        matches = bf.match(des_L[10], des_L[b])
        matches = sorted(matches, key=lambda x: x.distance)
        for item in matches[: 20]:
            featureIds_2.append(item.queryIdx)
    res_1, res_2 = Counter(featureIds_1), Counter(featureIds_2)
    queryIds_1, queryIds_2 = sorted(res_1, key=lambda x: res_1[x]), sorted(res_2, key=lambda x: res_2[x])
    chosen_ids_1, chosen_ids_2 = sorted(queryIds_1[-45:]), sorted(queryIds_2[-45:])

    print(chosen_ids_1,'\n',chosen_ids_2)  #100  [0, 2, 3,    7, 8, 13, 15, 25, 26, 27,     29, 30, 31, 32, 38, 41, 52, 54, 55, 56, 57, 62, 63, 65, 67, 68, 74, 75, 76, 77, 90, 91, 95, 96, 97, 104, 109, 110, 111, 112, 113, 115, 116, 125, 130]
                                           #1000 [0, 2, 3, 4, 7, 8, 13, 15, 25, 26, 27, 28, 29, 30, 31, 32, 40, 41, 52, 54, 55, 56, 57,     63, 65, 67, 68, 74, 75, 76, 77, 90,     95, 96, 97, 104, 109, 110, 111, 112, 113, 115, 116, 125, 130]
    sample_match = bf.match(des_L[0], des_L[10], None)  # match of frame 0 and 10
    sample_match = sorted(sample_match, key=lambda x: x.distance)

    trans_1,trans_2 = {}, {}  # set up the label translation
    for xx in range(45):
        trans_1[chosen_ids_1[xx]] = xx
    counter = 45
    for yy in range(45):
        for sample in sample_match[:70]:
            if chosen_ids_2[yy] == sample.trainIdx and sample.queryIdx in trans_1:
                trans_2[chosen_ids_2[yy]] = trans_1[sample.queryIdx]
        if chosen_ids_2[yy] not in trans_2:
            trans_2[chosen_ids_2[yy]] = counter
            counter += 1
    # print(trans_1, '\n',trans_2)

    for i in range(11, 511):
        match1, match2 = bf.match(des_L[0], des_L[i]), bf.match(des_L[10], des_L[i])  # bf.match(query, train)
        data_temp, labels_temp = [], []
        for pare1 in match1:
            dis_1 = math.sqrt((kp_L[0][pare1.queryIdx].pt[0] - kp_L[i][pare1.trainIdx].pt[0]) ** 2 + (kp_L[0][pare1.queryIdx].pt[1] - kp_L[i][pare1.trainIdx].pt[1]) ** 2)
            if pare1.queryIdx in chosen_ids_1 and dis_1 < 25:
                roi = cv.getRectSubPix(Images_L[i], (16, 16), kp_L[i][pare1.trainIdx].pt)
                data_temp.append(cv.cvtColor(roi, cv.COLOR_BGR2GRAY))
                labels_temp.append(trans_1[pare1.queryIdx])
        for pare2 in match2:
            dis_2 = math.sqrt((kp_L[10][pare2.queryIdx].pt[0] - kp_L[i][pare2.trainIdx].pt[0]) ** 2 + (kp_L[10][pare2.queryIdx].pt[1] - kp_L[i][pare2.trainIdx].pt[1]) ** 2)
            if pare2.queryIdx in chosen_ids_2 and dis_2 < 25 and trans_2[pare2.queryIdx] not in labels_temp:
                roi = cv.getRectSubPix(Images_L[i], (16, 16), kp_L[i][pare2.trainIdx].pt)
                data_temp.append(cv.cvtColor(roi, cv.COLOR_BGR2GRAY))
                labels_temp.append(trans_2[pare2.queryIdx])
        data.append(data_temp)
        labels.append(labels_temp)

    train_data, train_labels, eval_data, eval_labels = [], [], [], []  # training/evaluating data and label
    for roi_1 in data[:400]:
        train_data.extend(roi_1)
    for num_1 in labels[:400]:
        train_labels.extend(num_1)
    for roi_2 in data[400:]:
        eval_data.extend(roi_2)
    for num_2 in labels[400:]:
        eval_labels.extend(num_2)

    train_data = np.array(train_data, dtype=np.float32)
    eval_data = np.array(eval_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)
    eval_labels = np.array(eval_labels, dtype=np.int32)

    print('Train data amount : ', len(train_data), len(train_labels), '\nEvaluating data amount : ', len(eval_data), len(eval_labels))
    # np.savez('kp_data2.npz', train_data=train_data, train_labels=train_labels, eval_data=eval_data, eval_labels=eval_labels)

if __name__ == '__main__':
    main()
