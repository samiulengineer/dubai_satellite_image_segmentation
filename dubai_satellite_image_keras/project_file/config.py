""" config file is for hypermeter setup before running a model """

from datetime import datetime
import os



# GPU Selection
# ----------------------------------------------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"


# Image Input/Output
# ----------------------------------------------------------------------------------------------
height = 256
width = 256
in_channels = 3
num_classes = 6


# Training
# ----------------------------------------------------------------------------------------------
batch_size = 1
epochs = 1
learning_rate = 3e-4
model_name = "dncnn" # unet/mod-unet/dncnn/u2net


# Dataset
# ----------------------------------------------------------------------------------------------
patch_size = height # height = width, anyone is suitable
dataset_dir = "/home/mdsamiul/semantic-segmentation/data/Aerial_Image"
train_size = 0.8 
# valid_size = automatically set to 0.1
# test_size = automatically set to 0.1


# Logger/Callbacks
# ----------------------------------------------------------------------------------------------
tensorboard_log_name = "{}_epochs_{}_{}".format(model_name, epochs, datetime.now().strftime("%d-%b-%y"))
tensorboard_log_dir = "/home/mdsamiul/semantic-segmentation/dubai_satellite_image_keras/logs/{}/".format(model_name)

csv_log_name = "{}_epochs_{}_{}.csv".format(model_name, epochs, datetime.now().strftime("%d-%b-%y"))
csv_log_dir = "/home/mdsamiul/semantic-segmentation/dubai_satellite_image_keras/csv_logger/{}/".format(model_name)

checkpoint_name = "{}_epochs_{}_{}.hdf5".format(model_name, epochs, datetime.now().strftime("%d-%b-%y"))
checkpoint_dir = "/home/mdsamiul/semantic-segmentation/dubai_satellite_image_keras/model/{}/".format(model_name)

patience = 500 # required for early_stopping, if accuracy does not change for 500 epochs, model will stop automatically


# Evaluation
# ----------------------------------------------------------------------------------------------
load_model_name = "epochs_10000_dncnn_02-Jan-22.hdf5"
load_model_dir = "/home/mdsamiul/semantic-segmentation/dubai_satellite_image_keras/model/{}/".format(model_name)

test_img_number = 105
prediction_img_name = "test_img_{}".format(test_img_number)
prediction_dir = "/home/mdsamiul/semantic-segmentation/dubai_satellite_image_keras/prediction/{}/".format(model_name)