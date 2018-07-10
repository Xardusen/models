from collections import Counter
import math
import time
import numpy as np
import cv2 as cv
from collections import Counter
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


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

    logits = tf.layers.dense(inputs=dropout, units=68)  # input : [batch_size, 1024] / output : [batch_size, 45]
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
        img = cv.getRectSubPix(Images_Ls[number], (16, 16), point.pt)
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
    for q in range(68):
        predict_prob.append(0)
        predict_labels.append(0)
        predict_seq.append(0)
    classes = []
    for x in range(len(kp_L[number])):
        classes.append(predict[x]['classes'])
    classes = list(set(classes))

    for k in range(len(kp_L[number])):
        predict[k]["probabilities"] = max(predict[k]["probabilities"])
        for w in range(68):
            if predict[k]["classes"] == w and predict[k]["probabilities"] > predict_prob[w]:
                predict_seq[w] = k
                predict_labels[w] = predict[k]["classes"]
                predict_prob[w] = predict[k]["probabilities"]
    return  predict_seq, predict_labels, predict_prob, classes  # feature sequence in Image, feature label, feature probability, total classes


def match_on_cnn(pred_1, number_1, pred_2, number_2):

    # match12 = bf.match(des_L[0], des_L[1])
    # match12 = match12[:45]
    match12 = []
    for prm in range(68):
        init = cv.DMatch(0, 0, number_2)
        match12.append(init)

    for m in range(68):
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
        distance = pow(kp_L[number_1][ma_pre.queryIdx].pt[0] - kp_L[number_2][ma_pre.trainIdx].pt[0], 2) + pow(kp_L[number_1][ma_pre.queryIdx].pt[1] - kp_L[number_2][ma_pre.trainIdx].pt[1], 2)
        if distance > 25:
            match12_pre = match12_pre[:match12_pre.index(ma_pre)] + match12_pre[match12_pre.index(ma_pre) + 1 : ]
    return match12, match12_pre, float(len(match12_pre))/float(len(match12))  # return match, match after refining, precision


def video_to_images(filename, n): # function to convert video to images
    video = cv.VideoCapture(filename)
    # print('Height of video: ', video.get(cv.CAP_PROP_FRAME_HEIGHT), '\nWidth of video: ', video.get(cv.CAP_PROP_FRAME_WIDTH), '\nFrame of video: ', video.get(cv.CAP_PROP_FRAME_COUNT))
    total_images = []
    for ii in range(n):
        # video.set(cv.CAP_PROP_POS_FRAMES, n)
        total_images.append(video.read()[1])
    video.release()
    return total_images


time_start = time.clock()

Images_L = video_to_images("D:\QQBrowser\VideoData\\f5_dynamic_deint_L.avi", 1500)
kernel_size = []
for image in Images_L:
    kernel_size.append((3,3)) # 8.33(2) 8.48(3) 8.32(4) 8.14(5) for kernel_size arg
Images_Ls = list(map(cv.blur, Images_L, kernel_size))

orb = cv.ORB_create()
kp_L, des_L, kp_R, des_R = [], [], [], []  # store all features
for frame_L in Images_Ls:
    kp, des = orb.detectAndCompute(frame_L, None)
    kp_L.append(kp)
    des_L.append(des)

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

feature_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/feature_net_model/paper_vision1"
)  # create the classifier

tensor_to_log = {"probabilities": "softmax_tensor"}  # to show information while processing
logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=500)

# train_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"x": train_data},
#     y=train_labels,
#     batch_size=128,
#     num_epochs=None,
#     shuffle=True
# )
# feature_classifier.train(
#     input_fn=train_input_fn,
#     steps=100000,
#     # hooks=[logging_hook]
# )

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False
)
eval_results = feature_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)


time_mid = time.clock()

bf = cv.BFMatcher_create(cv.NORM_HAMMING)
sum_orb, sum_cnn, types_cnn = 0, 0, 0  # calculating acc
for test in range(600, 699):
    match_orb = bf.match(des_L[test], des_L[test + 1])
    match_orb_pre = match_orb.copy()
    for ma_pre in match_orb_pre:
        distance =pow(kp_L[test][ma_pre.queryIdx].pt[0] - kp_L[test + 1][ma_pre.trainIdx].pt[0], 2) + pow(kp_L[test][ma_pre.queryIdx].pt[1] - kp_L[test + 1][ma_pre.trainIdx].pt[1], 2)
        if distance > 25:
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
#     qq = cv.drawMatches(Images_L[testImage1], kp_L[testImage1], Images_L[testImage2], kp_L[testImage2], matchn[0], None, flags=2)
#     cv.imshow('CNN based', qq)
#     cv.waitKey(0)

# timing1 = time.clock()
# bf = cv.BFMatcher_create(cv.NORM_HAMMING)
# resultOne = []
# for test in range(1000):  # calculating running time
#     m_temp = bf.match(des_L[test], des_L[test + 1])
#     resultOne.append(m_temp)
# timing2 = time.clock()
# resultTwo = []
# for test2 in range(1000):
#     m_cnn = match_on_cnn(predict_on_cnn(test2), test2, predict_on_cnn(test +1), test +1)
#     resultTwo.append(m_cnn[0])
# timing3 = time.clock()
# print("orb 1000 frames : {}\ncnn 1000 frames : {}".format((timing2 - timing1), (timing3 - timing2)))

# time_end = time.clock()
# print('training time : {}\nevaluating time : {}'.format((time_mid - time_start), (time_end - time_mid)))
