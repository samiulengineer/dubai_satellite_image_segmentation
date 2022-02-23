""" config file is for hypermeter setup before running a model """

from datetime import datetime
import tensorflow as tf
import os



# GPU Selection
# ----------------------------------------------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], 
#                                                             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 4500)])
#   except RuntimeError as e:
#     print(e)


# Image Input/Output
# ----------------------------------------------------------------------------------------------
height = 256
width = 256
in_channels = 3
num_classes = 6


# Training
# ----------------------------------------------------------------------------------------------
model_name = "unet" # unet/mod-unet/dncnn/u2net/vnet/unet++
batch_size = 3
epochs = 3
learning_rate = 3e-4


# Dataset
# ----------------------------------------------------------------------------------------------
patch_size = height # height = width, anyone is suitable
dataset_dir = "/home/mdsamiul/semantic-segmentation/data/Aerial_Image"
train_size = 0.8 
# valid_size = automatically set to 0.1 for 0.8 train_size
# test_size = automatically set to 0.1  for 0.8 train_size

# Dataset
# ----------------------------------------------------------------------------------------------
"""Preprocessed dataset will save time because we do not need data preprocessing everytime"""

x_train_dir = "/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/x_train.npy"
x_valid_dir = "/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/x_valid.npy"
x_test_dir = "/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/x_test.npy"
y_train_dir = "/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/y_train.npy"
y_valid_dir = "/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/y_valid.npy"
y_test_dir = "/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/y_test.npy"
    

# Logger/Callbacks
# ----------------------------------------------------------------------------------------------
tensorboard_log_name = "{}_epochs_{}_{}".format(model_name, epochs, datetime.now().strftime("%d-%b-%y"))
tensorboard_log_dir = "/home/mdsamiul/semantic-segmentation/dubai_satellite_image_keras/logs/{}/".format(model_name)

csv_log_name = "{}_epochs_{}_{}.csv".format(model_name, epochs, datetime.now().strftime("%d-%b-%y"))
csv_log_dir = "/home/mdsamiul/semantic-segmentation/dubai_satellite_image_keras/csv_logger/{}/".format(model_name)

checkpoint_name = "{}_epochs_{}_{}.hdf5".format(model_name, epochs, datetime.now().strftime("%d-%b-%y"))
checkpoint_dir = "/home/mdsamiul/semantic-segmentation/dubai_satellite_image_keras/model/{}/".format(model_name)

early_stopping_technique = False
patience = 500 # required for early_stopping, if accuracy does not change for 500 epochs, model will stop automatically


# Evaluation
# ----------------------------------------------------------------------------------------------
load_model_name = "u2net_epochs_10000_13-Jan-22.hdf5"
load_model_dir = "/home/mdsamiul/semantic-segmentation/dubai_satellite_image_keras/model/{}/".format(model_name)

prediction_test_dir = "/home/mdsamiul/semantic-segmentation/dubai_satellite_image_keras/prediction/{}/test/".format(model_name)
prediction_val_dir = "/home/mdsamiul/semantic-segmentation/dubai_satellite_image_keras/prediction/{}/validation/".format(model_name)

single_image = True # if True, then only index x_test image will plot
index = 61