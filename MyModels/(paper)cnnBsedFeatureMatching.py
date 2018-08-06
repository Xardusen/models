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
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)  # input : [batch_size, 16, 16, 32] / output : [batch_size, 8, 8, 32]
    conv2 = tf.layers.conv2d(  # input : [batch_size, 8, 8, 32] / output : [batch_size, 8, 8, 64]
        inputs=pool1,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)  # input : [batch_size, 8, 8, 64] / output : [batch_size, 4, 4, 64]

    pool2_flat = tf.reshape(pool2, [-1, 4 * 4 * 128])  # input : [batch_size, 4, 4, 64] / output : [batch_size, 4 * 4 * 64]

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
        if predictionOne[1][seq] != -1 and predictionTwo[1][seq] != -1 :  # and predictionOne[2][seq] > 0.99 and predictionTwo[2][seq] > 0.99
            matchTemp = cv.DMatch(predictionOne[0][seq], predictionTwo[0][seq], numberTwo)
            matchOnCNN.append(matchTemp)

    refinedMatches = []
    for item in matchOnCNN:
        distance = (kp[numberOne][item.queryIdx].pt[0] - kp[numberTwo][item.trainIdx].pt[0]) ** 2 + \
                   (kp[numberOne][item.queryIdx].pt[1] - kp[numberTwo][item.trainIdx].pt[1]) ** 2
        if distance <= 25:
            refinedMatches.append(item)
    matchAccuracy = len(refinedMatches)/len(matchOnCNN)
    return matchOnCNN, refinedMatches, matchAccuracy  # return match, match after refining, precision


# ---------------------------------generate init images, keyPoints and descriptions -----------------------
images = video_to_images("D:\QQBrowser\VideoData\\f5_dynamic_deint_L.avi", 1500)
kernelSize = []
for __ in images:
    kernelSize.append((3, 3))
images = list(map(cv.blur, images, kernelSize))

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
for i in range(1, 200):
    matches = bf.match(des[0], des[i])
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
data, labels = [], []
data_acc = 0
for j in range(1, 201):
    dataTemp, labelsTemp = [], []
    matches = bf.match(des[0],des[j])
    for match in matches:
        if match.queryIdx in chosenIds:
            # distance = (kp[0][match.queryIdx].pt[0] - kp[j][match.trainIdx].pt[0]) ** 2 + \
            #            (kp[0][match.queryIdx].pt[1] - kp[j][match.trainIdx].pt[1]) ** 2
            if match.distance < 25:
                roi = cv.getRectSubPix(images[j], (16, 16), kp[j][match.trainIdx].pt)
                dataTemp.append(cv.cvtColor(roi, cv.COLOR_BGR2GRAY))
                labelsTemp.append(transition[match.queryIdx])
    data.append(dataTemp)
    labels.append(labelsTemp)

train_data, train_labels, eval_data, eval_labels = [], [], [], []  # training/evaluating data and label
cutPoint = int(len(data) * 0.8)
for roi_1 in data[:cutPoint]:
    train_data.extend(roi_1)
for num_1 in labels[:cutPoint]:
    train_labels.extend(num_1)
for roi_2 in data[cutPoint:]:
    eval_data.extend(roi_2)
for num_2 in labels[cutPoint:]:
    eval_labels.extend(num_2)
train_data = np.array(train_data, dtype=np.float32)
eval_data = np.array(eval_data, dtype=np.float32)
train_labels = np.array(train_labels, dtype=np.int32)
eval_labels = np.array(eval_labels, dtype=np.int32)

# ----------------------------------data,label amount / data accuracy -----------------------------------
print('Train data amount : ', len(train_data), len(train_labels), '\nEvaluating data amount : ',len\
    (eval_data), len(eval_labels))

# ------------------------------------------- cnn model ---------------------------------------
featureClassifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir="/temp/feature_net_model/cnnBased_final"
)
tensor_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=1000)

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

#  ---------------------------------------matching results evaluate ------------------------------------

sum_orb, sum_cnn = 0, 0
allMatch = len(kpCounter) * 20

for test in range(201, 301):
    matchCnn = match_on_cnn(predict_on_cnn(test), test, predict_on_cnn(test +1), test + 1)
    sum_cnn += len(matchCnn[1])
    print(test)
print("Matching accuracy on CNN :", sum_cnn/allMatch)

for i in range(201, 301):
    matches = bf.match(des[0], des[i])
    for match in matches:
        if match.queryIdx in chosenIds:
            # distance = (kp[0][match.queryIdx].pt[0] - kp[i][match.trainIdx].pt[0]) ** 2 + \
            #            (kp[0][match.queryIdx].pt[1] - kp[i][match.trainIdx].pt[1]) ** 2
            if match.distance < 25:
                sum_orb += 1
print("Matching accuracy on ORB :", sum_orb/allMatch)

# # CNN match image show
# while True:
#     num = int(input("Input Frame : "))
#     matchoncnn = match_on_cnn(predict_on_cnn(num), num, predict_on_cnn(num +1), num + 1)
#     matchimage = cv.drawMatches(images[num], kp[num], images[num + 1], kp[num + 1], matchoncnn[1], None, flags=2)
#     cv.imshow("Match on "+"{}".format(num), matchimage)
#     cv.waitKey(0)

# # orb match image show
# while True:
#     num = int(input("Input Frame : "))
#     matchonorb = bf.match(des[0], des[num])
#     refine = []
#     for matchorb in matchonorb:
#         if matchorb.queryIdx in chosenIds:
#             refine.append(matchorb)
#     matchimage = cv.drawMatches(images[0], kp[0], images[num], kp[num], refine, None, flags=2)
#     cv.imshow("Match on "+"{}".format(num), matchimage)
#     cv.waitKey(0)
