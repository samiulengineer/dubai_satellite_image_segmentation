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


# Dataset
# ----------------------------------------------------------------------------------------------
if (preprocessed_dataset == True):
    x_test = np.load(x_test_dir)
    y_test = np.load(y_test_dir)
else:
    from dataset import *


# Load Model
# ----------------------------------------------------------------------------------------------
model = load_model(os.path.join(load_model_dir, load_model_name), compile = False)


# Prediction on Test Dataset
# ----------------------------------------------------------------------------------------------
# ref- https://keras.io/examples/vision/oxford_pets_image_segmentation/
"""we can predict specific index image or predict all images together and save to a folder"""
"""predict same image for all models, we need to choose specific index number"""

"""used for confusion matrix"""
y_pred = model.predict(x_test)
y_pred_argmax = np.argmax(y_pred, axis = 3)

"""used for plot the prediction images"""
y_test_argmax = np.argmax(y_test, axis = 3)
# test_img_number = random.randint(0, len(x_test)) # taking a random index test image for testing
test_img_number = test_img_number # taking a fixed index test image for testing
ground_truth = y_test_argmax[test_img_number]

# test_img_norm = test_img[:,:,0][:,:,None]
test_img = x_test[test_img_number]

test_img_input = np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img = np.argmax(prediction, axis = 3)[0,:,:]


# Model Evaluation on Test Dataset
# ----------------------------------------------------------------------------------------------
metrics = ['acc',jacard_coef,precision_m,recall_m,f1_m,iou_coef,dice_coef,subset_accuracy,cat_acc]
model.compile(optimizer = "adam", loss = focal_loss(), metrics = metrics)
model.evaluate(x_test, y_test)


# Confusion Matrix
# ----------------------------------------------------------------------------------------------
cm = confusion_matrix(y_test_argmax.flatten(), y_pred_argmax.flatten())


if __name__ == '__main__':
    
    # Plot Predicted Image
    # ------------------------------------------------------------------------------------------
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img)
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth)
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img)
    plt.savefig(os.path.join(prediction_dir, prediction_img_name))
    
    
    # Plot Confusion Matrix
    # ----------------------------------------------------------------------------------------------
        
    plot_confusion_matrix(cm,
                          normalize = False,
                          target_names = ['Building', 'Land', 'Road', 'Vegetation', 'Water', 'Unlabeled'],
                          title = "Confusion Matrix")