import os
from metrics import *
from model import *
from loss import *
from dataset import *
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from datetime import datetime


# GPU Selection
# ----------------------------------------------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"


# Hyperparameters
# ----------------------------------------------------------------------------------------------
epochs = 10000
batch_size = 4
learning_rate = 3e-4


# Model
# ----------------------------------------------------------------------------------------------
model_name = "dncnn"

if(model_name == 'unet'):
    model = unet()
if(model_name == 'mod-unet'):
    model = mod_unet()
elif(model_name == 'dncnn'):
    model = DnCNN()
elif(model_name == 'u2net'):
    model = U2NET()
  
    
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
checkpoint= ModelCheckpoint(
    "/home/mdsamiul/InSAR-Coding/semantic_segmentation/dubai_satellite_image/model/epochs_{}".format(epochs) + "_{}".format(model_name)+ "_" + datetime.now().strftime("%d-%b-%y_%-I:%M:%S_%p" + ".hdf5"),
    save_best_only = True)


# Tensorbord Logger
# ----------------------------------------------------------------------------------------------
logdir = "/home/mdsamiul/InSAR-Coding/semantic_segmentation/dubai_satellite_image/logs/{}".format(model_name) + "_epochs_{}_".format(epochs) + datetime.now().strftime("%d-%b-%y_%-I:%M:%S_%p")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir = logdir)


# CSV Logger
# ----------------------------------------------------------------------------------------------
csv_logger = tf.keras.callbacks.CSVLogger("/home/mdsamiul/InSAR-Coding/semantic_segmentation/dubai_satellite_image/csv_logger/{}".format(model_name) + "_epochs_{}_".format(epochs) + datetime.now().strftime("%d-%b-%y_%-I:%M:%S_%p" + ".csv"), 
                                          separator = ",", 
                                          append = False)


# Early Stopping
# ----------------------------------------------------------------------------------------------
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='acc', patience=500)


# Prediction on Epoch
# ----------------------------------------------------------------------------------------------
# class PerformancePlotCallback(keras.callbacks.Callback):
#     def __init__(self, x_test, y_test, model_name):
#         self.x_test = x_test
#         self.y_test = y_test
#         self.model_name = model_name
        
#     def on_epoch_end(self, epoch, logs={}):
#         y_pred = self.model.predict(self.x_test)
#         fig, ax = plt.subplots(figsize=(8,4))
#         plt.scatter(y_test, y_pred, alpha=0.6, 
#             color='#FF0000', lw=1, ec='black')
        
#         lims = [0, 5]

#         plt.plot(lims, lims, lw=1, color='#0000FF')
#         plt.ticklabel_format(useOffset=False, style='plain')
#         plt.xticks(fontsize=18)
#         plt.yticks(fontsize=18)
#         plt.xlim(lims)
#         plt.ylim(lims)

#         plt.tight_layout()
#         plt.title(f'Prediction Visualization Keras Callback - Epoch: {epoch}')
#         plt.savefig('model_train_images/'+self.model_name+"_"+str(epoch))
#         plt.close()
        
        
        
#         # y_pred=self.model.predict(x_test)
#         # y_pred_argmax=np.argmax(y_pred, axis=3)

#         y_test_argmax=np.argmax(y_test, axis=3)
#         test_img_number = random.randint(0, len(x_test))

#         test_img = x_test[test_img_number]
#         ground_truth=y_test_argmax[test_img_number]

#         # test_img_norm=test_img[:,:,0][:,:,None]
#         test_img_input=np.expand_dims(test_img, 0)

#         prediction = (model.predict(test_img_input))
#         predicted_img=np.argmax(prediction, axis=3)[0,:,:]

        
#         plt.figure(figsize=(12, 8))
#         plt.subplot(231)
#         plt.title('Testing Image')
#         plt.imshow(test_img)
#         plt.subplot(232)
#         plt.title('Testing Label')
#         plt.imshow(ground_truth)
#         plt.subplot(233)
#         plt.title('Prediction on test image')
#         plt.imshow(predicted_img)
#         plt.title(f'Prediction Visualization Keras Callback - Epoch: {epoch}')
#         plt.savefig('model_train_images/'+self.model_name+"_"+str(epoch))
        
        

# prediction = PerformancePlotCallback(x_test, y_test, model_name)


# fit
# ----------------------------------------------------------------------------------------------
history = model.fit(x_train, y_train, 
                    batch_size = batch_size, 
                    verbose = 1, 
                    epochs = epochs,
                    validation_data = (x_test, y_test), 
                    shuffle = False,
                    # callbacks = [checkpoint, tensorboard_callback, csv_logger, early_stopping] # early_stopping included
                    callbacks = [checkpoint, tensorboard_callback, csv_logger]
                    )