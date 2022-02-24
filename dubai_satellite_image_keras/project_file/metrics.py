from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

        
      
def jacard_coef(y_true, y_pred, smooth = 100):
    intersection = K.sum(K.abs(y_true * y_pred), axis = -1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis = -1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall, 4

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def auc(y_true, y_pred):
    auc = tf.compat.v1.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.compat.v1.local_variables_initializer())
    return auc

def subset_accuracy(y_true, y_pred):
    threshold = tf.constant(.8, tf.float32)
    gtt_pred = tf.math.greater(y_pred, threshold)
    gtt_true = tf.math.greater(y_true, threshold)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(gtt_pred, gtt_true), tf.float32), axis=-1)
    return accuracy

def cat_acc(y_true, y_pred):
    return keras.metrics.categorical_accuracy(y_true,y_pred)

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

# Keras MeanIoU
# ----------------------------------------------------------------------------------------------

class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=3), tf.argmax(y_pred, axis=3), sample_weight)


# Matrics
# ----------------------------------------------------------------------------------------------

def get_metrics(config):
    """
    Summary:
        create keras MeanIoU object and all custom metrics dictornary
    Arguments:
        config (dict): configuration dictionary
    Return:
        metrics directories
    """

    
    m = MyMeanIOU(config['num_classes'])
    return {'jacard_coef':jacard_coef,
            'precision_m':precision_m,
            'recall_m':recall_m,
            'f1_m':f1_m,
            'iou_coef':iou_coef,
            'dice_coef':dice_coef,
            'subset_accuracy':subset_accuracy,
            'cat_acc':cat_acc,
            'MyMeanIOU': m
          }
#metrics = ['acc']