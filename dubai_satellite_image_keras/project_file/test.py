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
from metrics import jacard_coef, precision_m, recall_m, f1_m, iou_coef, dice_coef, subset_accuracy, cat_acc
from loss import focal_loss
from config import *
from utils import prediction, pred_plot, plot_confusion_matrix
import pathlib


# Print Experimental Setup before Training
# ----------------------------------------------------------------------------------------------
print("Load Model = {}".format(load_model_name))
print("Preprocessed Data = {}".format(os.path.exists(x_test_dir)))


# Model Output Path
# ----------------------------------------------------------------------------------------------
pathlib.Path(prediction_test_dir).mkdir(parents = True, exist_ok = True)


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


# Prediction
# ----------------------------------------------------------------------------------------------
feature, mask, pred_mask = prediction(index, x_test, y_test, model)


# Prediction Plot
# ----------------------------------------------------------------------------------------------
total_pred_img = 5 # len(x_test) for all images or any ineteger value less than the length of the x_test
# prediction_name = "test_img_{}_acc_{:.4f}.png".format(index, eval[1])

if (single_image):
    feature, mask, pred_mask = prediction(index, x_test, y_test, model)
    pred_plot(feature, mask, pred_mask, index, prediction_test_dir, model, x_test, y_test)
else : 
    for index in range(total_pred_img):
        feature, mask, pred_mask = prediction(index, x_test, y_test, model)
        pred_plot(feature, mask, pred_mask, index, prediction_test_dir, model, x_test, y_test)



# Average Metrics Score on Test Dataset
# ----------------------------------------------------------------------------------------------
metrics = ['acc', jacard_coef, precision_m, recall_m, f1_m, iou_coef, dice_coef, subset_accuracy, cat_acc]
model.compile(optimizer = "adam", loss = focal_loss(), metrics = metrics)
eval = model.evaluate(x_test, y_test)


# Confusion Matrix
# ----------------------------------------------------------------------------------------------
cm = confusion_matrix(np.argmax(y_test, axis = 3).flatten(), # y_true
                      np.argmax(model.predict(x_test), axis = 3).flatten()) # y_pred

plot_confusion_matrix(cm,
                    normalize = False,
                    target_names = ['Building', 'Land', 'Road', 'Vegetation', 'Water', 'Unlabeled'],
                    title = "Confusion Matrix")