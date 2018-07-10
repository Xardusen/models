import cv2 as cv
from Kp_Data_Generator import video_to_images
from collections import Counter
import math
import tensorflow as tf
import numpy as np


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

    logits = tf.layers.dense(inputs=dropout, units=len(chosenIds))  # input : [batch_size, 1024] / output : [batch_size, 45]
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
    for point in kp[number]:
        img = cv.getRectSubPix(images[number], (16, 16), point.pt)
        predict_data.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    predict_data = np.array(predict_data, dtype=np.float32)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": predict_data},
        y=None,
        num_epochs=1,
        shuffle=False
    )
    predict = featureClassifier.predict(input_fn=predict_input_fn)
    predict = list(result for result in predict)

    predict_seq, predict_prob, predict_labels = [], [], []
    for q in range(68):
        predict_prob.append(0)
        predict_labels.append(0)
        predict_seq.append(0)
    classes = []
    for x in range(len(kp[number])):
        classes.append(predict[x]['classes'])
    classes = list(set(classes))

    for k in range(len(kp[number])):
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
        distance = pow(kp[number_1][ma_pre.queryIdx].pt[0] - kp[number_2][ma_pre.trainIdx].pt[0], 2) + pow(kp[number_1][ma_pre.queryIdx].pt[1] - kp[number_2][ma_pre.trainIdx].pt[1], 2)
        if distance > 25:
            match12_pre = match12_pre[:match12_pre.index(ma_pre)] + match12_pre[match12_pre.index(ma_pre) + 1 : ]
    return match12, match12_pre, float(len(match12_pre))/float(len(match12))  # return match, match after refining, precision


# ---------------------------------generate init images, keyPoints and descriptions -----------------------
images = video_to_images("D:\QQBrowser\VideoData\\f5_dynamic_deint_L.avi", 1000)
kernelSize = []
for __ in images:
    kernelSize.append((3, 3))
blurredImage = list(map(cv.blur, images, kernelSize))

orb = cv.ORB_create()
kp, des = [], []
for image in images:
    kpTemp, desTemp = orb.detectAndCompute(image, None)
    kp.append(kpTemp)
    des.append(desTemp)

#  --------------------------- calculate transition -----------------------------
kpIds, chosenIds  = [], []
bf = cv.BFMatcher_create(cv.NORM_HAMMING, crossCheck=True)
transition = {}
for i in range(1, 100):
    matches = bf.match(des[0], des[i])
    matches = sorted(matches, key=lambda x:x.distance)
    for match in matches:
        kpIds.append(match.queryIdx)
kpCounter = Counter(kpIds)
kpMostCommon = kpCounter.most_common(int(len(kpCounter)/5))
labelCounter = 0
for kpId in kpMostCommon:
    chosenIds.append(kpId[0])
    transition[kpId[0]] = labelCounter
    labelCounter += 1

# --------------------------------generate data and labels ---------------------------------
data, labels, accuracy = [], [], []
for j in range(1, 101):
    dataTemp, labelsTemp = [], []
    total, correct = 0, 0
    matches = bf.match(des[0],des[j])
    for match in matches:
        if match.queryIdx in chosenIds:
            roi = cv.getRectSubPix(images[j], (16, 16), kp[j][match.trainIdx].pt)
            dataTemp.append(cv.cvtColor(roi, cv.COLOR_BGR2GRAY))
            labelsTemp.append(transition[match.queryIdx])
            total += 1
            dist = math.sqrt((kp[0][match.queryIdx].pt[0] - kp[j][match.trainIdx].pt[0]) ** 2 + \
                             (kp[0][match.queryIdx].pt[1] - kp[j][match.trainIdx].pt[1]) ** 2)
            if dist < 16:
                correct += 1
    data.append(dataTemp)
    labels.append(labelsTemp)
    accuracy.append(correct/total)

train_data, train_labels, eval_data, eval_labels = [], [], [], []
for rect in data[:80]:
    train_data.extend(rect)
for label in labels[:80]:
    train_labels.extend(label)
for rect2 in data[80:]:
    eval_data.extend(rect2)
for label2 in labels[80:]:
    eval_labels.extend(label2)

train_data = np.array(train_data, dtype=np.float32)
eval_data = np.array(eval_data, dtype=np.float32)
train_labels = np.array(train_labels, dtype=np.int32)
eval_labels = np.array(eval_labels, dtype=np.int32)

# ----------------------------------data,label amount / data accuracy -----------------------------------
print('Train data amount : ', len(train_data), len(train_labels), '\nEvaluating data amount : ',len\
    (eval_data), len(eval_labels))
averageAccuracy = 0
for x in accuracy:
    averageAccuracy += x
print('Data and labels accuracy :', averageAccuracy/100)

# ------------------------------------------- cnn model ---------------------------------------
featureClassifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/temp/feature_net_model/cnnBased_1")

trainInputFn = tf.estimator.inputs.numpy_input_fn(
    x={"x" : train_data},
    y=train_labels,
    batch_size=128,
    num_epochs=None,
    shuffle=True
)
featureClassifier.train(
    input_fn=trainInputFn,
    steps=30000,
)

evalInputFn = tf.estimator.inputs.numpy_input_fn(
    x={"x" : eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False
)
evalResult = featureClassifier.evaluate(input_fn=evalInputFn)
print(evalResult)

#  ---------------------------------------matching result evaluate ------------------------------------

sum_orb, sum_cnn, types_cnn = 0, 0, 0  # calculating acc
for test in range(600, 699):
    # match_orb = bf.match(des[test], des[test + 1])
    # match_orb_pre = match_orb.copy()
    # for ma_pre in match_orb_pre:
    #     distance =pow(kp[test][ma_pre.queryIdx].pt[0] - kp[test + 1][ma_pre.trainIdx].pt[0], 2) + pow(kp[test][ma_pre.queryIdx].pt[1] - kp[test + 1][ma_pre.trainIdx].pt[1], 2)
    #     if distance > 25:
    #         match_orb_pre = match_orb_pre[:match_orb_pre.index(ma_pre)] + match_orb_pre[match_orb_pre.index(ma_pre) + 1:]
    # sum_orb += float(len(match_orb_pre))/float(len(match_orb))

    match_cnn = match_on_cnn(predict_on_cnn(test), test, predict_on_cnn(test + 1), test + 1)
    sum_cnn += match_cnn[2]
    types_cnn += len(predict_on_cnn(test)[3])
    print(test)
print(sum_orb/99, sum_cnn/99, types_cnn/99)