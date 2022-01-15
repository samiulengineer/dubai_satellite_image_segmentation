from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from loss import focal_loss

# Prediction
# ----------------------------------------------------------------------------------------------
def prediction(index, x_test, y_test, model):
    """
    predict image ndarray based on feature and mask
    """
    feature = x_test[index]
    mask = np.argmax(y_test, axis = 3)[index]

    test_img_input = np.expand_dims(feature, axis = 0)
    prediction = (model.predict(test_img_input))
    pred_mask = np.argmax(prediction, axis = 3)[0,:,:]
    
    return feature, mask, pred_mask


# Plot the prediction
# ----------------------------------------------------------------------------------------------
def pred_plot(feature, mask, pred_mask, index, prediction_dir, model, x_test, y_test):
    
    """[save feature, mask & predicted images as a subplot with prediction accuracy]

    Args:
        feature (numpy): [description]
        mask (numpy): [description]
        pred_mask (numpy): [description]
    """
    
    metrics = ['acc']
    model.compile(optimizer = "adam", loss = focal_loss(), metrics = metrics)
    eval = model.evaluate(x_test[index:index + 1], y_test[index:index + 1]) # evaluate s specific index test/valid image
    
    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.title("Feature")
    plt.imshow(feature)

    plt.subplot(232)
    plt.title("Mask")
    plt.imshow(mask)

    plt.subplot(233)
    plt.title("Prediction (Accuracy_{:.4f})".format(eval[1]))
    plt.imshow(pred_mask)
    plt.tight_layout()

    prediction_name = "test_img_{}_acc_{:.4f}.png".format(index, eval[1])
    return plt.savefig(os.path.join(prediction_dir, prediction_name), bbox_inches='tight')


# Plot Confusion Matrix
# ----------------------------------------------------------------------------------------------
def plot_confusion_matrix(cm,
                          target_names,
                          title = 'Confusion matrix',
                          cmap = None,
                          normalize = True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
    
