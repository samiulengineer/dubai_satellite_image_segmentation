import os 
import cv2
import random
import numpy as np
import tensorflow 
from matplotlib import pyplot as plt
from patchify import patchify
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from PIL import Image
from tensorflow.keras.utils import to_categorical
import os 
from sklearn.model_selection import train_test_split
from config import *
from fast_ml.model_development import train_valid_test_split






image_dataset = []

for path, subdirs, files in os.walk(dataset_dir, topdown=True):
    # print(sorted(subdirs))
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'images':   #Find all 'images' directories
        images = os.listdir(path)#List of all image names in this subdirectory
        images = sorted(images)
        for i, image_name in enumerate(images):  
            
            if image_name.endswith(".jpg"):   #Only read jpg images...
            
                image = cv2.imread(path+"/"+image_name, 1)  #Read each image as BGR
                SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
                SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
                image = Image.fromarray(image)
                image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                # image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                image = np.array(image)             
    
                #Extract patches from each image
                print("Now patchifying image:", path+"/"+image_name)
                patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
        
                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                        
                        single_patch_img = patches_img[i,j,:,:]
                        
                        #Use minmaxscaler instead of just dividing by 255. 
                        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                        
                        #single_patch_img = (single_patch_img.astype('float32')) / 255. 
                        single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                        image_dataset.append(single_patch_img)
                        
image_dataset = np.array(image_dataset)
                        
                        
mask_dataset = []

for path, subdirs, files in os.walk(dataset_dir, topdown=True):
        
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'masks':   #Find all 'images' directories
        masks = os.listdir(path)  #List of all image names in this subdirectory
        masks = sorted(masks)
        for i, mask_name in enumerate(masks): 

            if mask_name.endswith(".png"):   #Only read png images... (masks in this dataset)
            
                mask = cv2.imread(path+"/"+mask_name, 1)  #Read each image as Grey (or color but remember to map each color to an integer)
                mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                SIZE_X = (mask.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
                SIZE_Y = (mask.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
                mask = Image.fromarray(mask)
                mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                # mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                mask = np.array(mask)             
    
                #Extract patches from each image
                print("Now patchifying mask:", path+"/"+mask_name)
                patches_mask = patchify(mask, (patch_size, patch_size, 3), step = patch_size)  #Step=256 for 256 patches means no overlap
        
                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        
                        single_patch_mask = patches_mask[i,j,:,:]
                        #single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                        single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.                               
                        mask_dataset.append(single_patch_mask)

mask_dataset = np.array(mask_dataset)                            
                        
                        
Building = '#3C1098'.lstrip('#')
Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152

Land = '#8429F6'.lstrip('#')
Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246

Road = '#6EC1E4'.lstrip('#') 
Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228

Vegetation =  'FEDD3A'.lstrip('#') 
Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58

Water = 'E2A929'.lstrip('#') 
Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41

Unlabeled = '#9B9B9B'.lstrip('#') 
Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155



def rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    label_seg = np.zeros(label.shape,dtype = np.uint8)
    label_seg [np.all(label == Building, axis = -1)] = 0
    label_seg [np.all(label == Land, axis = -1)] = 1
    label_seg [np.all(label == Road, axis = -1)] = 2
    label_seg [np.all(label == Vegetation, axis = -1)] = 3
    label_seg [np.all(label == Water, axis = -1)] = 4
    label_seg [np.all(label == Unlabeled, axis = -1)] = 5
    
    label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg


labels = []

for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)

labels = np.array(labels)

labels = np.expand_dims(labels, axis = 3)

n_classes = len(np.unique(labels))
labels_cat = to_categorical(labels, num_classes=n_classes)


# Input data Splitting
# ----------------------------------------------------------------------------------------------
def data_split():
    x_train, x_rem, y_train, y_rem = train_test_split(image_dataset, labels_cat, train_size = train_size)
    x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size = 0.5)
    return x_train, y_train, x_valid, y_valid, x_test, y_test

x_train, y_train, x_valid, y_valid, x_test, y_test = data_split()




if __name__ == '__main__':
    
    # mask_dataset =  np.array(mask_dataset)
    # image_dataset = np.array(image_dataset)
    
    """save training, validation and test data and labels separately,
       It will save time to load the original data everytime,
       model can run directly from the saved dataset,
       when the data preprocessing changes, we need to save the preprocessed data again
    """

    # np.save("/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/x_train.npy", x_train)
    # np.save("/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/x_valid.npy", x_valid)
    # np.save("/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/x_test.npy", x_test)
    # np.save("/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/y_train.npy", y_train)
    # np.save("/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/y_valid.npy", y_valid)
    # np.save("/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/y_test.npy", y_test)

    # x_train = np.load("/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/x_train.npy")
    # x_valid = np.load("/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/x_valid.npy")
    # x_test = np.load("/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/x_test.npy")
    # y_train = np.load("/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/y_train.npy")
    # y_valid = np.load("/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/y_valid.npy")
    # y_test = np.load("/home/mdsamiul/semantic-segmentation/data/Aerial_Image/preprocessed_data/y_test.npy")
    
    
    print("Total number of images : {}".format(len(image_dataset)))
    print("Total number of masks : {}".format(len(mask_dataset)))
    
    print("x_train shape : {}".format(x_train.shape))
    print("x_valid shape : {}".format(x_valid.shape))
    print("x_test shape : {}".format(x_test.shape))
    
    print("y_train shape : {}".format(y_train.shape))
    print("y_valid shape : {}".format(y_valid.shape))
    print("y_test shape : {}".format(y_test.shape))
    
    print(labels_cat.shape) # all 6 classes
    image_number = random.randint(0, len(image_dataset))
    print(image_number)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(np.reshape(image_dataset[image_number], (patch_size, patch_size, 3)))
    plt.subplot(122)
    plt.imshow(np.reshape(mask_dataset[image_number], (patch_size, patch_size, 3)))
    plt.show()
    
   
    
    print("Unique labels in label dataset are: ", np.unique(label))

    image_number = random.randint(0, len(image_dataset))
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(image_dataset[image_number])
    plt.subplot(122)
    plt.imshow(labels[image_number][:,:,0])
    plt.show()