import numpy as np
import cv2 as cv
from Kp_Data_Generator import video_to_images
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

kp_data = np.load('kp_data2.npz')
train_data = kp_data['train_data']
train_labels = kp_data['train_labels']
eval_data = kp_data['eval_data']
eval_labels = kp_data['eval_labels']

print('Train data amount : ', len(train_data), '\nEvaluating data amount : ', len(eval_data))
print(len(set(eval_labels)))
feature_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/feature_net_model/100_frames_100_v3"
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
#     steps=50000,
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

# bf = cv.BFMatcher_create(cv.NORM_HAMMING)
# sum_orb, sum_cnn, types_cnn = 0, 0, 0  # calculating acc
# for test in range(1200, 1299):
#     match_orb = bf.match(des_L[test], des_L[test + 1])
#     match_orb_pre = match_orb.copy()
#     for ma_pre in match_orb_pre:
#         distance =pow(kp_L[test][ma_pre.queryIdx].pt[0] - kp_L[test + 1][ma_pre.trainIdx].pt[0], 2) + pow(kp_L[test][ma_pre.queryIdx].pt[1] - kp_L[test + 1][ma_pre.trainIdx].pt[1], 2)
#         if distance > 25:
#             match_orb_pre = match_orb_pre[:match_orb_pre.index(ma_pre)] + match_orb_pre[match_orb_pre.index(ma_pre) + 1:]
#     sum_orb += float(len(match_orb_pre))/float(len(match_orb))
#
#     match_cnn = match_on_cnn(predict_on_cnn(test), test, predict_on_cnn(test + 1), test + 1)
#     sum_cnn += match_cnn[2]
#     types_cnn += len(predict_on_cnn(test)[3])
#     print(test)
# print(sum_orb/99, sum_cnn/99, types_cnn/99)


while True:  # test on input certain frames
    testImage1 = int(input('first frame : '))
    testImage2 = int(input('second frame : '))
    prediction_1 = predict_on_cnn(testImage1)  # aa = [predict_seq, predict_labels, predict_prob, classes]
    prediction_2 = predict_on_cnn(testImage2)
    matchn = match_on_cnn(prediction_1, testImage1, prediction_2, testImage2)
    print(len(matchn[0]), len(matchn[1]), matchn[2])
    qq = cv.drawMatches(Images_L[testImage1], kp_L[testImage1], Images_L[testImage2], kp_L[testImage2], matchn[0], None)
    cv.imshow('1', qq)
    cv.waitKey(0)
