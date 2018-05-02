import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from collections import Counter
import tensorflow as tf
import math

tf.logging.set_verbosity(tf.logging.INFO)
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


def cnn_model_fn(features, labels, mode):  # model_fn for classifier
    input_layer = tf.reshape(features["x"], [-1, 16, 16, 1])
    conv1 = tf.layers.conv2d(  # input : [batch_size, 8, 8, 1] / output : [batch_size, 8, 8, 32]
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)  # input : [batch_size, 16, 16, 32] / output : [batch_size, 8, 8, 32]
    conv2 = tf.layers.conv2d(  # input : [batch_size, 8, 8, 32] / output : [batch_size, 8, 8, 64]
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)  # input : [batch_size, 8, 8, 64] / output : [batch_size, 4, 4, 64]
    pool2_flat = tf.reshape(pool2, [-1, 4* 4* 64])  # input : [batch_size, 4, 4, 64] / output : [batch_size, 4 * 4 * 64]

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)  # input : [batch_size, 4 * 4 * 64] / output : [batch_size, 1024]
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=45)  # input : [batch_size, 1024] / output : [batch_size, 45]
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


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
for frame_l in Images_Ls:
    kp, des = orb.detectAndCompute(frame_l, None)
    kp_L.append(kp)
    des_L.append(des)
for frame_r in Images_Rs:
    kp, des = orb.detectAndCompute(frame_r, None)
    kp_R.append(kp)
    des_R.append(des)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True) # use brute force matcher
data, labels_all = [], []  # train and evaluate sources

total_features = []  # calculate the chosen_ids
for b in range(998):
    matches = bf.match(des_L[0], des_L[b + 1])
    matches = sorted(matches, key=lambda  x: x.distance)
    for item in matches[: 10]:
        total_features.append(item.queryIdx)
res = Counter(total_features)
query_xs = sorted(res, key=lambda x: res[x])
chosen_ids = sorted(query_xs[-45:])  # query_ids of kp_L[0] in 1000 frames (kp which shows over 80 times)
# print('total Counter result : ', res, '\n', 'result length : ', len(res), '\n', 'chosen_ids : ', chosen_ids)

for i in range(998):
    matches = bf.match(des_L[0], des_L[i + 1])  # bf.match(querydes, traindes)
    matches = sorted(matches, key=lambda x: x.distance)
    data_temp, labels_temp = [], []

    for match in matches[: 10]:
        if match.queryIdx in chosen_ids:
            roi = cv.getRectSubPix(Images_L[i + 1], (16, 16), kp_L[i + 1][match.trainIdx].pt)
            data_temp.append(cv.cvtColor(roi, cv.COLOR_BGR2GRAY))
            labels_temp.append(match.queryIdx)

    data.append(data_temp)
    labels_all.append(labels_temp)


translation = {}  # convert labels to [0:45]
for i in range(45):
    translation[chosen_ids[i]] = i

labels2 = []
for label in labels_all:
    label2 = [translation[x] if x in translation else x for x in label]
    labels2.append(label2)

train_data, train_labels, eva_data, eva_labels = [], [], [], []  # training/evaluating data and label
for roi_1 in data[:800]:
    train_data.extend(roi_1)
for num_1 in labels2[:800]:
    train_labels.extend(num_1)
for roi_2 in data[800:]:
    eva_data.extend(roi_2)
for num_2 in labels2[800:]:
    eva_labels.extend(num_2)
print(len(train_labels), len(train_data), len(eva_labels), len(eva_data))

train_data = np.array(train_data, dtype=np.float32)
eva_data = np.array(eva_data, dtype=np.float32)

train_labels = np.array(train_labels, dtype=np.int32)
eva_labels = np.array(eva_labels, dtype=np.int32)


feature_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/feature_net_model/1000_frames_bs_200"
)  # create the classifier

tensor_to_log = {"probabilities": "softmax_tensor"}  # to show information while processing
logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=500)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=200,
    num_epochs=None,
    shuffle=True
)
feature_classifier.train(
    input_fn=train_input_fn,
    steps=3000,
    # hooks=[logging_hook]
)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eva_data},
    y=eva_labels,
    num_epochs=1,
    shuffle=False
)
eval_results = feature_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)

# predict_data = []  # frame 101 processing
# for point in kp_L[100]:
#     img = cv.getRectSubPix(Images_L[100], (8, 8), point.pt)
#     predict_data.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
# predict_data = np.array(predict_data, dtype=np.float32)
#
# predict_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"x": predict_data},
#     y=None,
#     num_epochs=1,
#     shuffle=False
# )
# predict = feature_classifier.predict(input_fn=predict_input_fn)
# predict = list(item for item in predict)
#
# predict_seq, predict_prob, predict_labels = [], [], []
# for q in range(15):
#     predict_prob.append(0)
#     predict_labels.append(0)
#     predict_seq.append(0)
#
# for k in range(len(kp_L[100])):
#     predict[k]["probabilities"] = max(predict[k]["probabilities"])
#     for w in range(15):
#         if predict[k]["classes"] == w and predict[k]["probabilities"] > predict_prob[w]:
#             predict_seq[w] = k
#             predict_labels[w] = predict[k]["classes"]
#             predict_prob[w] = predict[k]["probabilities"]
#
# # print(predict_seq,'\n', predict_labels, '\n', predict_prob)
#
# predict_data_1 = []  # frame 102 processing
# for point in kp_L[101]:
#     img = cv.getRectSubPix(Images_L[101], (8, 8), point.pt)
#     predict_data_1.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
# predict_data_1 = np.array(predict_data_1, dtype=np.float32)
#
# predict_input_fn_1 = tf.estimator.inputs.numpy_input_fn(
#     x={"x": predict_data_1},
#     y=None,
#     num_epochs=1,
#     shuffle=False
# )
# predict_1 = feature_classifier.predict(input_fn=predict_input_fn_1)
# predict_1 = list(item for item in predict_1)
#
# predict_seq_1, predict_prob_1, predict_labels_1 = [], [], []
# for q in range(15):
#     predict_prob_1.append(0)
#     predict_labels_1.append(0)
#     predict_seq_1.append(0)
#
# for k in range(len(kp_L[101])):
#     predict_1[k]["probabilities"] = max(predict_1[k]["probabilities"])
#     for w in range(15):
#         if predict_1[k]["classes"] == w and predict_1[k]["probabilities"] > predict_prob_1[w]:
#             predict_seq_1[w] = k  # used while matching
#             predict_labels_1[w] = predict_1[k]["classes"]
#             predict_prob_1[w] = predict_1[k]["probabilities"]
#
# # print(predict_seq,'\n', predict_labels, '\n', predict_prob)
# # print('\n', predict_seq_1,'\n', predict_labels_1, '\n', predict_prob_1)
# match_101_102 = bf.match(des_L[100], des_L[101], None)
#
# location = []
# for m in range(15):
#     match_101_102[m].queryIdx = predict_seq[m]
#     match_101_102[m].trainIdx = predict_seq_1[m]
#     location.append(math.sqrt(pow(kp_L[100][predict_seq[m]].pt[0]-kp_L[101][predict_seq_1[m]].pt[0], 2)+pow(kp_L[100][predict_seq[m]].pt[1]-kp_L[101][predict_seq_1[m]].pt[1], 2)))
#
# ma101_2 = cv.drawMatches(Images_L[100], kp_L[100], Images_L[101], kp_L[101], match_101_102[12:13], None, flags=2)
#
# # cv.imshow('1', ma101_2)
# print(location)
# cv.waitKey(0)
