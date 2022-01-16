import os
import math
import pathlib
from loss import *
from model import *
from config import *
from metrics import *
from tensorflow import keras
from utils import prediction, pred_plot
from tensorflow.keras.callbacks import ModelCheckpoint


# Print Experimental Setup before Training
# ----------------------------------------------------------------------------------------------
print("Model = {}".format(model_name))
print("Epochs = {}".format(epochs))
print("Batch Size = {}".format(batch_size))
print("Preprocessed Data = {}".format(os.path.exists(x_train_dir)))


# Model Output Path
# ----------------------------------------------------------------------------------------------
pathlib.Path(csv_log_dir).mkdir(parents = True, exist_ok = True)
pathlib.Path(tensorboard_log_dir).mkdir(parents = True, exist_ok = True)
pathlib.Path(checkpoint_dir).mkdir(parents = True, exist_ok = True)
pathlib.Path(prediction_val_dir).mkdir(parents = True, exist_ok = True)


# Dataset
# ----------------------------------------------------------------------------------------------
if (os.path.exists(x_train_dir)):
    x_train = np.load(x_train_dir)
    x_valid = np.load(x_valid_dir)
    y_train = np.load(y_train_dir)
    y_valid = np.load(y_valid_dir)
else:
    from dataset import *
    

# Model
# ----------------------------------------------------------------------------------------------
if(model_name == 'unet'):
    model = unet(num_classes = num_classes, img_height = height, img_width = width, in_channels = in_channels)
elif(model_name == 'mod-unet'):
    model = mod_unet(num_classes = num_classes, img_height = height, img_width = width, in_channels = in_channels)
elif(model_name == 'dncnn'):
    model = DnCNN(num_classes = num_classes, img_height = height, img_width = width, in_channels = in_channels)
elif(model_name == 'u2net'):
    model = u2net(num_classes = num_classes, img_height = height, img_width = width, in_channels = in_channels)
elif(model_name == 'vnet'):
    model = vnet(num_classes = num_classes, img_height = height, img_width = width, in_channels = in_channels)
elif(model_name == 'unet++'):
    model = unet_plus_plus(num_classes = num_classes, img_height = height, img_width = width, in_channels = in_channels)
  

# Optimizer
# ----------------------------------------------------------------------------------------------
adam = keras.optimizers.Adam(learning_rate = learning_rate)


# Compile
# ----------------------------------------------------------------------------------------------
model.compile(optimizer = adam, loss = focal_loss(), metrics = metrics)


# checkpoint
# ----------------------------------------------------------------------------------------------
checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, checkpoint_name), save_best_only = True)


# Tensorbord Logger
# ----------------------------------------------------------------------------------------------
tensorboard_callback = keras.callbacks.TensorBoard(log_dir = os.path.join(tensorboard_log_dir, tensorboard_log_name))


# CSV Logger
# ----------------------------------------------------------------------------------------------
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(csv_log_dir, csv_log_name), separator = ",", append = False)


# Early Stopping
# ----------------------------------------------------------------------------------------------
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'acc', patience = patience)


# Learning Rate Scheduler
# ----------------------------------------------------------------------------------------------
"""learning rate decrease according to the model performance"""
def lr_scheduler(epoch):
    drop = 0.5
    epoch_drop = epochs / 8.
    lr = learning_rate * math.pow(drop, math.floor((1 + epoch) / epoch_drop))
    return lr

lr_decay = tf.keras.callbacks.LearningRateScheduler(schedule = lr_scheduler)


# Prediction during Training
# ----------------------------------------------------------------------------------------------
class PerformancePlotCallback(keras.callbacks.Callback):
    
    """[save a prediction image after each epoch with accuracy]
    Args:
        x_valid = validation features
        y_valid = validation masks
        model = model object
    Return:
        plot feature, mask, validation maks after every epoch
    """

    def __init__(self, x_valid, y_valid, model):
        super(keras.callbacks.Callback, self).__init__()
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.model = model
        
    def on_epoch_end(self, epoch, logs={}):
        if (epoch % 2 == 0): # every after certain epochs the model will predict mask
            feature, mask, pred_mask = prediction(index, self.x_valid, self.y_valid, self.model)
            
            """below two line code import pred_plot from utils but shows keyerror - "cat_acc"""""
            # prediction_name = "test" # this is the error point where name can not pass
            # pred_plot(feature, mask, pred_mask, index, prediction_dir, prediction_name, model, x_test, y_test)
            
            # metrics = ['acc']
            # model.compile(optimizer = "adam", loss = focal_loss(), metrics = metrics) # 2bd time compile is not required
            eval = model.evaluate(x_valid[index:index + 1], y_valid[index:index + 1])

            plt.figure(figsize=(12, 8))
            
            plt.subplot(231)
            plt.title("Feature")
            plt.imshow(feature)
            
            plt.subplot(232)
            plt.title("Mask")
            plt.imshow(mask)
            
            plt.subplot(233)
            # plt.title("Prediction")
            plt.title("Prediction (Accuracy_{:.4f})".format(eval[1]))
            plt.imshow(pred_mask)
            plt.tight_layout()
            
            plt.savefig(os.path.join(prediction_val_dir, "trn_img_{}-epoch_{}".format(index, epoch)), bbox_inches='tight')
        
            
pred_during_training = PerformancePlotCallback(x_valid, y_valid, model) 


# Callbacks
# ----------------------------------------------------------------------------------------------
if (early_stopping_technique):
    callbacks = [checkpoint, tensorboard_callback, csv_logger, early_stopping, lr_decay, pred_during_training]
else:
    # callbacks = [checkpoint, tensorboard_callback, csv_logger, lr_decay, pred_during_training]
    callbacks = [checkpoint, tensorboard_callback, csv_logger, lr_decay]


# fit
# ----------------------------------------------------------------------------------------------
history = model.fit(x_train, y_train, 
                    batch_size = batch_size, 
                    verbose = 1, 
                    epochs = epochs,
                    validation_data = (x_valid, y_valid), 
                    shuffle = False,
                    callbacks = callbacks,
                    )