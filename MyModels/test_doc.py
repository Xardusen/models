import cv2 as cv
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


def computeDistance(p1, p2):
    sums = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    dist = math.sqrt(sums)
    return  dist


Images = video_to_images("D:\QQBrowser\VideoData\\f5_dynamic_deint_L.avi", 10)
brisk = cv.BRISK_create()
orb = cv.ORB_create()
bf = cv.BFMatcher_create(cv.NORM_HAMMING, True)

kp_1, des_1 = orb.detectAndCompute(Images[0], None)
kp_2, des_2 = orb.detectAndCompute(Images[1], None)

matches = bf.match(des_1, des_2)
matches_s= []
# print(kp_1[match[0].queryIdx].pt[0], kp_1[match[0].queryIdx].pt[1])
for match in matches:
    temp_dist = computeDistance(kp_1[match.queryIdx].pt, kp_2[match.trainIdx].pt)
    if 4 < temp_dist < 5:
        matches_s.append(match)
print(len(matches), len(matches_s))

counter = 0
num = 0
while counter < len(matches_s):
    print(num)
    ima = cv.drawMatches(Images[0], kp_1, Images[1], kp_2, matches_s[num : num + 1], None, flags=2)
    cv.imshow('11', ima)
    cv.waitKey(0)
    counter += 1
    num += 1
