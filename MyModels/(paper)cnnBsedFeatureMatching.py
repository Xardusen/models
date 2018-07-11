import cv2 as cv
from Kp_Data_Generator import video_to_images
from collections import Counter
import math
import tensorflow as tf
import numpy as np


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
    predictData = []
    for point in kp[number]:
        img = cv.getRectSubPix(images[number], (16, 16), point.pt)
        predictData.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    predictData = np.array(predictData, dtype=np.float32)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": predictData},
        y=None,
        num_epochs=1,
        shuffle=False
    )
    predict = featureClassifier.predict(input_fn=predict_input_fn)
    predict = list(predict)

    predictSequence, predictLabel, predictProbability = [], [], []
    for __ in range(len(chosenIds)):
        predictSequence.append(-1)
        predictLabel.append(-1)
        predictProbability.append(-1)

    for kpNum in range(len(kp[number])):
        kpProbability = max(predict[kpNum]["probabilities"])
        kpClass = predict[kpNum]["classes"]
        if kpProbability > predictProbability[kpClass]:
            predictSequence[kpClass] = kpNum
            predictLabel[kpClass] = kpClass
            predictProbability[kpClass] = kpProbability
    return  predictSequence, predictLabel, predictProbability  # feature sequence in Image, feature label, feature probability


def match_on_cnn(predictionOne, numberOne, predictionTwo, numberTwo):
    matchOnCNN = []
    for seq in range(len(chosenIds)):
        if predictionOne[1][seq] != -1 and predictionTwo[1][seq] != -1 and predictionOne[2][seq] > 0.99 and predictionTwo[2][seq] > 0.99:
            matchTemp = cv.DMatch(predictionOne[0][seq], predictionTwo[0][seq], numberTwo)
            matchOnCNN.append(matchTemp)

    refinedMatches = []
    for item in matchOnCNN:
        distance = (kp[numberOne][item.queryIdx].pt[0] - kp[numberTwo][item.trainIdx].pt[0]) ** 2 + (kp[numberOne][item.queryIdx].pt[1] - kp[numberTwo][item.trainIdx].pt[1]) ** 2
        if distance <= 25:
            refinedMatches.append(item)
    matchAccuracy = len(refinedMatches)/len(matchOnCNN)
    return matchOnCNN, refinedMatches, matchAccuracy  # return match, match after refining, precision


# ---------------------------------generate init images, keyPoints and descriptions -----------------------
images = video_to_images("D:\QQBrowser\VideoData\\f5_dynamic_deint_L.avi", 1500)
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
tensor_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=500)

# trainInputFn = tf.estimator.inputs.numpy_input_fn(
#     x={"x" : train_data},
#     y=train_labels,
#     batch_size=128,
#     num_epochs=None,
#     shuffle=True
# )
# featureClassifier.train(
#     input_fn=trainInputFn,
#     steps=30000,
# )

evalInputFn = tf.estimator.inputs.numpy_input_fn(
    x={"x" : eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False
)
evalResult = featureClassifier.evaluate(input_fn=evalInputFn)
print(evalResult)

#  ---------------------------------------matching result evaluate ------------------------------------

# sum_orb, sum_cnn = 0, 0  # calculating acc
# for test in range(101, 201):
#     matchORB = bf.match(des[test], des[test + 1])
#     counter = 0
#     for matchO in matchORB:
#         distance =(kp[test][matchO.queryIdx].pt[0] - kp[test + 1][matchO.trainIdx].pt[0]) ** 2 + (kp[test][matchO.queryIdx].pt[1] - kp[test + 1][matchO.trainIdx].pt[1]) ** 2
#         if distance <= 25:
#             counter += 1
#     sum_orb += counter/len(matchORB)
#
#     matchCnn = match_on_cnn(predict_on_cnn(test), test, predict_on_cnn(test + 1), test + 1)
#     sum_cnn += matchCnn[2]
#     print(test)
# print(sum_orb/100, sum_cnn/100)

