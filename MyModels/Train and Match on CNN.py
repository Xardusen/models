import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from collections import Counter
import tensorflow as tf
import math

tf.logging.set_verbosity(tf.logging.INFO)
'''
In this file we get the feature data from matcher of the Image_L[0] with the rest of
Image_L in the first 1000 frames 
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


def predict_on_cnn(number):  # Compute feature and classes(probability) on a frame
    predict_data = []
    for point in kp_L[number]:
        img = cv.getRectSubPix(Images_L[number], (16, 16), point.pt)
        predict_data.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    predict_data = np.array(predict_data, dtype=np.float32)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": predict_data},
        y=None,
        num_epochs=1,
        shuffle=False
    )
    predict = feature_classifier.predict(input_fn=predict_input_fn)
    predict = list(result for result in predict)

    predict_seq, predict_prob, predict_labels = [], [], []
    for q in range(45):
        predict_prob.append(0)
        predict_labels.append(0)
        predict_seq.append(0)
    classes = []
    for x in range(len(kp_L[number])):
        classes.append(predict[x]['classes'])
    classes = list(set(classes))

    for k in range(len(kp_L[number])):
        predict[k]["probabilities"] = max(predict[k]["probabilities"])
        for w in range(45):
            if predict[k]["classes"] == w and predict[k]["probabilities"] > predict_prob[w]:
                predict_seq[w] = k
                predict_labels[w] = predict[k]["classes"]
                predict_prob[w] = predict[k]["probabilities"]
    return  predict_seq, predict_labels, predict_prob, classes  # feature sequence in Image, feature label, feature probability, total classes


def match_on_cnn(pred_1, number_1, pred_2, number_2):

    # match12 = bf.match(des_L[0], des_L[1])
    # match12 = match12[:45]
    match12 = []
    for prm in range(45):
        init = cv.DMatch(0, 0, number_2)
        match12.append(init)

    for m in range(45):
        # distance = math.sqrt(pow(kp_L[number_1][pred_1[0][m]].pt[0] - kp_L[number_2][pred_2[0][m]].pt[0], 2) + pow(kp_L[number_1][pred_1[0][m]].pt[1] - kp_L[number_2][pred_2[0][m]].pt[1], 2))
        # if distance < 20:
            if pred_1[2][m] > 0.9:
                match12[m].queryIdx = pred_1[0][m]
            else:
                match12[m].queryIdx = -1
            if pred_2[2][m] > 0.9:
                match12[m].trainIdx = pred_2[0][m]
            else:
                match12[m].trainIdx = -1
        # else:
        #     match12[m].queryIdx = -1
        #     match12[m].trainIdx = -1
    for ma in match12:
        if ma.queryIdx == -1 or ma.trainIdx == -1:
            match12 = match12[ : match12.index(ma)] + match12[match12.index(ma) + 1 : ]
            # match12.remove(ma)

    match12_pre = match12.copy()
    for ma_pre in match12_pre:
        distance = math.sqrt(pow(kp_L[number_1][ma_pre.queryIdx].pt[0] - kp_L[number_2][ma_pre.trainIdx].pt[0], 2) + pow(kp_L[number_1][ma_pre.queryIdx].pt[1] - kp_L[number_2][ma_pre.trainIdx].pt[1], 2))
        if distance > 5:
            match12_pre = match12_pre[:match12_pre.index(ma_pre)] + match12_pre[match12_pre.index(ma_pre) + 1 : ]
    return match12, match12_pre, float(len(match12_pre))/float(len(match12))  # return match, match after refining, precision


Images_L = video_to_images("D:\QQBrowser\VideoData\\f5_dynamic_deint_L.avi", 1500) # read video
Images_R = video_to_images("D:\QQBrowser\VideoData\\f5_dynamic_deint_R.avi", 1500)

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
print('Train data amount : ', len(train_data), '\nEvaluating data amount : ', len(eva_data))

train_data = np.array(train_data, dtype=np.float32)
eva_data = np.array(eva_data, dtype=np.float32)

train_labels = np.array(train_labels, dtype=np.int32)
eva_labels = np.array(eva_labels, dtype=np.int32)


feature_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/feature_net_model/1000_frames_use30000"
)  # create the classifier

tensor_to_log = {"probabilities": "softmax_tensor"}  # to show information while processing
logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=500)

# train_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"x": train_data},
#     y=train_labels,
#     batch_size=100,
#     num_epochs=None,
#     shuffle=True
# )
# feature_classifier.train(
#     input_fn=train_input_fn,
#     steps=30000,
#     # hooks=[logging_hook]
# )

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eva_data},
    y=eva_labels,
    num_epochs=1,
    shuffle=False
)
eval_results = feature_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)

bf2 = cv.BFMatcher_create(cv.NORM_HAMMING)
sum_orb, sum_cnn, types_cnn = 0, 0, 0  # calculating acc
for test in range(1000, 1099):
    match_orb = bf2.match(des_L[test], des_L[test + 1])
    match_orb_pre = match_orb.copy()
    for ma_pre in match_orb_pre:
        distance = math.sqrt(pow(kp_L[test][ma_pre.queryIdx].pt[0] - kp_L[test + 1][ma_pre.trainIdx].pt[0], 2) + pow(kp_L[test][ma_pre.queryIdx].pt[1] - kp_L[test + 1][ma_pre.trainIdx].pt[1], 2))
        if distance > 5:
            match_orb_pre = match_orb_pre[:match_orb_pre.index(ma_pre)] + match_orb_pre[match_orb_pre.index(ma_pre) + 1:]
    sum_orb += float(len(match_orb_pre))/float(len(match_orb))

    match_cnn = match_on_cnn(predict_on_cnn(test), test, predict_on_cnn(test + 1), test + 1)
    sum_cnn += match_cnn[2]
    types_cnn += len(predict_on_cnn(test)[3])
    print(test)
print(sum_orb/99, sum_cnn/99, types_cnn/99)


# while True:  # test on input certain frames
#     testImage1 = int(input('first frame : '))
#     testImage2 = int(input('second frame : '))
#     prediction_1 = predict_on_cnn(testImage1)  # aa = [predict_seq, predict_labels, predict_prob, classes]
#     prediction_2 = predict_on_cnn(testImage2)
#     matchn = match_on_cnn(prediction_1, testImage1, prediction_2, testImage2)
#     print(len(matchn[0]), len(matchn[1]), matchn[2])
#     qq = cv.drawMatches(Images_L[testImage1], kp_L[testImage1], Images_L[testImage2], kp_L[testImage2], matchn[0], None)
#     cv.imshow('1', qq)
#     cv.waitKey(0)
