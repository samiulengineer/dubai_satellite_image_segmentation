import random
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import itertools
from metrics import plot_confusion_matrix, jacard_coef, precision_m, recall_m, f1_m, iou_coef, dice_coef, subset_accuracy, cat_acc
from loss import focal_loss
from config import *


# Print Experimental Setup before Training
# ----------------------------------------------------------------------------------------------
print("Load Model = {}".format(load_model_name))
print("Preprocessed Data = {}".format(os.path.exists(x_test_dir)))


# Dataset
# ----------------------------------------------------------------------------------------------
if (os.path.exists(x_test_dir)):
    x_test = np.load(x_test_dir)
    y_test = np.load(y_test_dir)
else:
    from dataset import *


# Load Model
# ----------------------------------------------------------------------------------------------
model = load_model(os.path.join(load_model_dir, load_model_name), compile = False)


# Prediction on Test Dataset
# ----------------------------------------------------------------------------------------------
# for num in range(len(x_test)):
for num in range(5):
    metrics = ['acc']
    model.compile(optimizer = "adam", loss = focal_loss(), metrics = metrics)
    eval = model.evaluate(x_test[num:num+1], y_test[num:num+1])
    
    test_img = x_test[num]

    y_test_argmax = np.argmax(y_test, axis = 3)
    ground_truth = y_test_argmax[num]

    test_img_input = np.expand_dims(test_img, axis = 0)
    prediction = (model.predict(test_img_input))
    predicted_img = np.argmax(prediction, axis = 3)[0,:,:]

    plt.figure(figsize=(12, 8))
    
    plt.subplot(231)
    plt.title("Feature")
    plt.imshow(test_img)
    
    plt.subplot(232)
    plt.title("Mask")
    plt.imshow(ground_truth)
    
    plt.subplot(233)
    plt.title("Prediction (Accuracy_{})".format(round(eval[1], 4)))
    plt.imshow(predicted_img)
    plt.tight_layout()
    
    metrics = ['acc']
    model.compile(optimizer = "adam", loss = focal_loss(), metrics = metrics)
    eval = model.evaluate(x_test[num:num+1], y_test[num:num+1])
    
    plt.savefig(os.path.join(prediction_dir, "test_img_{}_acc_{}.png".format(num, round(eval[1], 4))), bbox_inches='tight')



# Average Metrics Score on Test Dataset
# ----------------------------------------------------------------------------------------------
metrics = ['acc',jacard_coef,precision_m,recall_m,f1_m,iou_coef,dice_coef,subset_accuracy,cat_acc]
model.compile(optimizer = "adam", loss = focal_loss(), metrics = metrics)
eval = model.evaluate(x_test, y_test)


# Confusion Matrix
# ----------------------------------------------------------------------------------------------
y_pred = model.predict(x_test)
y_pred_argmax = np.argmax(y_pred, axis = 3)

cm = confusion_matrix(y_test_argmax.flatten(), y_pred_argmax.flatten())

plot_confusion_matrix(cm,
                    normalize = False,
                    target_names = ['Building', 'Land', 'Road', 'Vegetation', 'Water', 'Unlabeled'],
                    title = "Confusion Matrix")