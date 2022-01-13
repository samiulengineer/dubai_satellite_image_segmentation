import os
from metrics import *
from model import *
from loss import *
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from datetime import datetime
from config import *
import math


# Print Experimental Setup before Training
# ----------------------------------------------------------------------------------------------
print("Model = {}".format(model_name))
print("Epochs = {}".format(epochs))
print("Batch Size = {}".format(batch_size))
print("Preprocessed Data = {}".format(os.path.exists(x_train_dir)))


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
  
    
# Metrices
# ----------------------------------------------------------------------------------------------
metrics = ['acc',jacard_coef,precision_m,recall_m,f1_m,iou_coef,dice_coef,subset_accuracy,cat_acc]


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
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'acc', patience = patience)


# Learning Rate Scheduler
# ----------------------------------------------------------------------------------------------
"""learning rate decrease according to the model performance"""
def lr_scheduler(epoch):
    drop = 0.5
    epoch_drop = epochs / 8.
    lr = learning_rate * math.pow(drop, math.floor((1 + epoch) / epoch_drop))
    return lr

lr_decay = tf.keras.callbacks.LearningRateScheduler(schedule = lr_scheduler)


# Prediction on Epoch
# ----------------------------------------------------------------------------------------------
class PerformancePlotCallback(keras.callbacks.Callback):
    def __init__(self, x_valid, y_valid, model):
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.model = model
        
    def on_epoch_end(self, epoch, logs={}):
        valid_img = self.x_valid
        print(valid_img.shape)
        
        y_valid_argmax = np.argmax(self.y_valid, axis = 3)
        ground_truth = y_valid_argmax
        print(ground_truth.shape)

        valid_img_input = np.expand_dims(valid_img, axis = 0)
        prediction = (self.model.predict(valid_img_input))
        predicted_img = np.argmax(prediction, axis = 3)[0,:,:]

   
        plt.figure(figsize=(12, 8))
        
        plt.subplot(231)
        plt.title("Feature")
        plt.imshow(valid_img)
        
        plt.subplot(232)
        plt.title("Mask")
        plt.imshow(ground_truth)
        
        plt.subplot(233)
        plt.title("Prediction")
        plt.imshow(predicted_img)
        plt.tight_layout()
            
performance = PerformancePlotCallback(x_valid, y_valid, model)    
        


# fit
# ----------------------------------------------------------------------------------------------
history = model.fit(x_train, y_train, 
                    batch_size = batch_size, 
                    verbose = 1, 
                    epochs = epochs,
                    validation_data = (x_valid, y_valid), 
                    shuffle = False,
                    # callbacks = [checkpoint, tensorboard_callback, csv_logger, early_stopping, lr_decay] # early_stopping included
                    callbacks = [checkpoint, tensorboard_callback, csv_logger, lr_decay, performance]
                    )