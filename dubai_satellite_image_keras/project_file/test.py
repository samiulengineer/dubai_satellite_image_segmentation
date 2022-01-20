import os
import pathlib
import numpy as np
from config import *
from metrics import *
import tensorflow as tf
from loss import focal_loss
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from utils import prediction, pred_plot, plot_confusion_matrix
from metrics import jacard_coef, precision_m, recall_m, f1_m, iou_coef, dice_coef, subset_accuracy, cat_acc


# Print Experimental Setup before Training
# ----------------------------------------------------------------------------------------------
print("Model Name = {}".format(model_name))
print("Load Model = {}".format(load_model_name))
print("Preprocessed Data = {}".format(os.path.exists(x_test_dir)))


# Model Output Path
# ----------------------------------------------------------------------------------------------
"""if the prediction test directory is not available, create the directory; otherwise pass"""
pathlib.Path(prediction_test_dir).mkdir(parents = True, exist_ok = True)


# Dataset
# ----------------------------------------------------------------------------------------------
"""if the preprocessed dataset is available as npy, load from them
    otherwise, do all data preprocessing"""
if (os.path.exists(x_test_dir)):
    x_test = np.load(x_test_dir, mmap_mode='r')
    y_test = np.load(y_test_dir, mmap_mode='r')
else:
    from dataset import *


# Load Model
# ----------------------------------------------------------------------------------------------
"""load model from load_model_dir, load_model_name & model_name
   model_name is included inside the load_model_dir"""
model = load_model(os.path.join(load_model_dir, load_model_name), compile = False)


# Prediction Plot
# ----------------------------------------------------------------------------------------------
"""plot single image prediction or all images from x_test"""
total_pred_img = len(x_test) # len(x_test) for all images or any ineteger value less than the length of the x_test

if (single_image):
    feature, mask, pred_mask = prediction(index, x_test, y_test, model)
    pred_plot(feature, mask, pred_mask, index, prediction_test_dir, model, x_test, y_test)
else : 
    for index in range(total_pred_img):
        feature, mask, pred_mask = prediction(index, x_test, y_test, model)
        pred_plot(feature, mask, pred_mask, index, prediction_test_dir, model, x_test, y_test)


# Average Metrics Score on Test Dataset
# ----------------------------------------------------------------------------------------------
model.compile(optimizer = "adam", loss = focal_loss(), metrics = metrics)
eval = model.evaluate(x_test, y_test)


# Confusion Matrix
# ----------------------------------------------------------------------------------------------
"""flatten y_test and y_pred to plot confusion matrix"""
# y_true_flatten = np.argmax(y_test, axis = 3).flatten()
# y_pred_flatten = np.argmax(model.predict(x_test), axis = 3).flatten()

# cm = confusion_matrix(y_true_flatten, y_pred_flatten)

# plot_confusion_matrix(cm,
#                     normalize = False,
#                     target_names = ['Building', 'Land', 'Road', 'Vegetation', 'Water', 'Unlabeled'],
#                     title = "Confusion Matrix")


# Same Image prediction by all models
# ----------------------------------------------------------------------------------------------
"""load model, all hdf5 models should be saved in the same environment"""
# unet_model = load_model("/home/mdsamiul/semantic-segmentation/dubai_satellite_image_keras/model/unet/unet_epochs_100000_09-Jan-22.hdf5", compile = False)
# mod_unet_model = load_model("/home/mdsamiul/semantic-segmentation/dubai_satellite_image_keras/model/unet++/unet++_epochs_10000_11-Jan-22.hdf5", compile = False)

"""model prediction for all models"""
# feature = x_test[index]
# mask = np.argmax(y_test, axis = 3)[index]

# test_img_input = np.expand_dims(feature, axis = 0)
# prediction = (unet_model.predict(test_img_input))
# pred_mask = np.argmax(prediction, axis = 3)[0,:,:]

# metrics = ['acc']
# unet_model.compile(optimizer = "adam", loss = focal_loss(), metrics = metrics)
# eval_unet = unet_model.evaluate(x_test[index:index + 1], y_test[index:index + 1]) # evaluate specific index test/valid image

# mod_unet_model.compile(optimizer = "adam", loss = focal_loss(), metrics = metrics)
# eval_mod_unet = mod_unet_model.evaluate(x_test[index:index + 1], y_test[index:index + 1]) # evaluate specific index test/valid image

"""plot the model prediction"""
# plt.figure(figsize=(12, 8))

# plt.subplot(241)
# plt.title("Feature")
# plt.imshow(feature)

# plt.subplot(242)
# plt.title("Mask")
# plt.imshow(mask)

# plt.subplot(243)
# plt.title("UNET Prediction (Accuracy_{:.4f})".format(eval_unet[1]))
# plt.imshow(pred_mask)

# plt.subplot(244)
# plt.title("MOD-UNET Prediction (Accuracy_{:.4f})".format(eval_mod_unet[1]))
# plt.imshow(pred_mask)

# plt.savefig("test.png", bbox_inches='tight')