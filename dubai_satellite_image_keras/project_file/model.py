import os
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, ReLU, Concatenate, Activation, MaxPool2D, Lambda
from config import *


# UNET Model
# ----------------------------------------------------------------------------------------------

def unet(num_classes = num_classes, img_height = height, img_width = width, in_channels = in_channels):
    
    inputs = Input((img_height, img_width, in_channels))
 
    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.2)(c1)  # Original 0.1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)  # Original 0.1
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)  # Original 0.1
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model
    
    
    
# Modification UNET Model
# ----------------------------------------------------------------------------------------------

def mod_unet(num_classes = num_classes, img_height = height, img_width = width, in_channels = in_channels):
    
    inputs = Input((img_height, img_width, in_channels))
    
    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.2)(c1)  # Original 0.1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.2)(c5)  # Original 0.1
    c5 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    p5 = MaxPooling2D((2, 2))(c5)
     
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p5)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    p6 = MaxPooling2D((2, 2))(c6)
     
    c7 = Conv2D(1012, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p6)
    c7 = Dropout(0.3)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    # Expansive path 
    
    u8 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c6])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c5])
    c9 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        
    u10 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c9)
    u10 = concatenate([u10, c4])
    c10 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u10)
    c10 = Dropout(0.2)(c10)
    c10 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c10)
     
    u11 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c10)
    u11 = concatenate([u11, c3])
    c11 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u11)
    c11 = Dropout(0.2)(c11)
    c11 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c11)
     
    u12 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c11)
    u12 = concatenate([u12, c2])
    c12 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u12)
    c12 = Dropout(0.2)(c12)  # Original 0.1
    c12 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c12)
     
    u13 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c12)
    u13 = concatenate([u13, c1], axis=3)
    c13 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u13)
    c13 = Dropout(0.2)(c13)  # Original 0.1
    c13 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c13)
     
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c13)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model



# U2Net Model
# ----------------------------------------------------------------------------------------------

def basicblocks(input, filter, dilates = 1):
    x1 = Conv2D(filter, (3, 3), padding = 'same', dilation_rate = 1*dilates)(input)
    x1 = ReLU()(BatchNormalization()(x1))
    return x1

def RSU7(input, in_ch = 3, mid_ch = 12, out_ch = 3):
    hx = input
    #1
    hxin = basicblocks(hx, out_ch, 1)
    hx1 = basicblocks(hxin, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides = 2)(hx1)
    #2
    hx2 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides = 2)(hx2)
    #3
    hx3 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides = 2)(hx3)
    #4
    hx4 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides = 2)(hx4)
    #5
    hx5 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides = 2)(hx5)
    #6
    hx6 = basicblocks(hx, mid_ch, 1)
    #7
    hx7 = basicblocks(hx6, mid_ch, 2)

    #down
    #6
    hx6d = Concatenate(axis = -1)([hx7, hx6])
    hx6d = basicblocks(hx6d, mid_ch, 1)
    a,b,c,d = K.int_shape(hx5)
    hx6d=keras.layers.UpSampling2D(size=(2,2))(hx6d)

    #5
    hx5d = Concatenate(axis=-1)([hx6d, hx5])
    hx5d = basicblocks(hx5d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx4)
    hx5d = keras.layers.UpSampling2D(size=(2,2))(hx5d)

    # 4
    hx4d = Concatenate(axis=-1)([hx5d, hx4])
    hx4d = basicblocks(hx4d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx3)
    hx4d = keras.layers.UpSampling2D(size=(2,2))(hx4d)

    # 3
    hx3d = Concatenate(axis=-1)([hx4d, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx2)
    hx3d = keras.layers.UpSampling2D(size=(2,2))(hx3d)

    # 2
    hx2d = Concatenate(axis=-1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx1)
    hx2d = keras.layers.UpSampling2D(size=(2,2))(hx2d)

    # 1
    hx1d = Concatenate(axis=-1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output=keras.layers.add([hx1d,hxin])
    return output

def RSU6(input,in_ch=3,mid_ch=12,out_ch=3):
    hx=input
    #1
    hxin=basicblocks(hx,out_ch,1)
    hx1=basicblocks(hxin,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx1)
    #2
    hx2=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx2)
    #3
    hx3 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx3)
    #4
    hx4=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx4)
    #5
    hx5 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx5)
    #6
    hx6=basicblocks(hx,mid_ch,1)
    hx6=keras.layers.UpSampling2D((2, 2))(hx6)

    #5
    hx5d = Concatenate(axis=-1)([hx6, hx5])
    hx5d = basicblocks(hx5d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx4)
    hx5d = keras.layers.UpSampling2D(size=(2,2))(hx5d)

    # 4
    hx4d = Concatenate(axis=-1)([hx5d, hx4])
    hx4d = basicblocks(hx4d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx3)
    hx4d = keras.layers.UpSampling2D(size=(2,2))(hx4d)

    # 3
    hx3d = Concatenate(axis=-1)([hx4d, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx2)
    hx3d = keras.layers.UpSampling2D(size=(2,2))(hx3d)

    # 2
    hx2d = Concatenate(axis=-1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx1)
    hx2d = keras.layers.UpSampling2D(size=(2,2))(hx2d)

    # 1
    hx1d = Concatenate(axis=-1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output=keras.layers.add([hx1d,hxin])
    return output

def RSU5(input,in_ch=3,mid_ch=12,out_ch=3):
    hx=input
    #1
    hxin=basicblocks(hx,out_ch,1)
    hx1=basicblocks(hxin,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx1)
    #2
    hx2=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx2)
    #3
    hx3 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx3)
    #4
    hx4=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx4)
    #5
    hx5 = basicblocks(hx, mid_ch, 1)
    #hx5 = keras.layers.MaxPool2D((2, 2), strides=2)(hx5)
    hx5 = keras.layers.UpSampling2D((2, 2))(hx5)
    # 4
    hx4d = Concatenate(axis=-1)([hx5, hx4])
    hx4d = basicblocks(hx4d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx3)
    hx4d = keras.layers.UpSampling2D(size=(2,2))(hx4d)

    # 3
    hx3d = Concatenate(axis=-1)([hx4d, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx2)
    hx3d = keras.layers.UpSampling2D(size=(2,2))(hx3d)

    # 2
    hx2d = Concatenate(axis=-1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx1)
    hx2d = keras.layers.UpSampling2D(size=(2,2))(hx2d)

    # 1
    hx1d = Concatenate(axis=-1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output=keras.layers.add([hx1d,hxin])
    return output

def RSU4(input,in_ch=3,mid_ch=12,out_ch=3):
    hx=input
    #1
    hxin=basicblocks(hx,out_ch,1)
    hx1=basicblocks(hxin,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx1)
    #2
    hx2=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx2)
    #3
    hx3 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx3)
    #4
    hx4=basicblocks(hx,mid_ch,1)
    hx4=keras.layers.UpSampling2D((2,2))(hx4)

    # 3
    hx3d = Concatenate(axis=-1)([hx4, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx2)
    hx3d = keras.layers.UpSampling2D(size=(2,2))(hx3d)

    # 2
    hx2d = Concatenate(axis=-1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx1)
    hx2d = keras.layers.UpSampling2D(size=(2,2))(hx2d)

    # 1
    hx1d = Concatenate(axis=-1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output=keras.layers.add([hx1d,hxin])
    return output

def RSU4f(input, in_ch = 3, mid_ch = 12, out_ch = 3):
    hx=input
    #1
    hxin = basicblocks(hx, out_ch, 1)
    hx1 = basicblocks(hxin, mid_ch, 1)
    #2
    hx2=basicblocks(hx, mid_ch, 2)
    #3
    hx3 = basicblocks(hx, mid_ch, 4)
    #4
    hx4=basicblocks(hx, mid_ch, 8)

    # 3
    hx3d = Concatenate(axis = -1)([hx4, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 4)

    # 2
    hx2d = Concatenate(axis = -1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 2)

    # 1
    hx1d = Concatenate(axis = -1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output = keras.layers.add([hx1d, hxin])
    return output


def u2net(img_height = height, img_width = width, in_channels = in_channels, num_classes = num_classes):

    input = Input((img_height, img_width, in_channels))

    stage1 = RSU7(input, in_ch = 3, mid_ch = 32, out_ch = 64)
    stage1p = keras.layers.MaxPool2D((2,2), strides = 2)(stage1)

    stage2 = RSU6(stage1p, in_ch = 64, mid_ch = 32, out_ch = 128)
    stage2p = keras.layers.MaxPool2D((2, 2), strides = 2)(stage2)

    stage3 = RSU5(stage2p, in_ch = 128, mid_ch = 64, out_ch = 256)
    stage3p = keras.layers.MaxPool2D((2, 2), strides = 2)(stage3)

    stage4 = RSU4(stage3p, in_ch = 256, mid_ch = 128, out_ch = 512)
    stage4p = keras.layers.MaxPool2D((2, 2), strides = 2)(stage4)

    stage5 = RSU4f(stage4p, in_ch = 512, mid_ch = 256, out_ch = 512)
    stage5p = keras.layers.MaxPool2D((2, 2), strides = 2)(stage5)

    stage6 = RSU4f(stage5, in_ch = 512, mid_ch = 256, out_ch = 512)
    stage6u = keras.layers.UpSampling2D((1, 1))(stage6)

    #decoder
    stage6a = Concatenate(axis = -1)([stage6u,stage5])
    stage5d = RSU4f(stage6a, 1024, 256, 512)
    stage5du = keras.layers.UpSampling2D((2, 2))(stage5d)

    stage5a = Concatenate(axis = -1)([stage5du, stage4])
    stage4d = RSU4(stage5a, 1024, 128, 256)
    stage4du = keras.layers.UpSampling2D((2, 2))(stage4d)

    stage4a = Concatenate(axis = -1)([stage4du, stage3])
    stage3d = RSU5(stage4a, 512, 64, 128)
    stage3du = keras.layers.UpSampling2D((2, 2))(stage3d)

    stage3a = Concatenate(axis = -1)([stage3du, stage2])
    stage2d = RSU6(stage3a, 256, 32, 64)
    stage2du = keras.layers.UpSampling2D((2, 2))(stage2d)

    stage2a = Concatenate(axis = -1)([stage2du, stage1])
    stage1d = RSU6(stage2a, 128, 16, 64)

    #side output
    side1 = Conv2D(num_classes, (3, 3), padding = 'same', name = 'side1')(stage1d)
    side2 = Conv2D(num_classes, (3, 3), padding = 'same')(stage2d)
    side2 = keras.layers.UpSampling2D((2, 2), name = 'side2')(side2)
    side3 = Conv2D(num_classes, (3, 3), padding = 'same')(stage3d)
    side3 = keras.layers.UpSampling2D((4, 4), name = 'side3')(side3)
    side4 = Conv2D(num_classes, (3, 3), padding = 'same')(stage4d)
    side4 = keras.layers.UpSampling2D((8, 8), name = 'side4')(side4)
    side5 = Conv2D(num_classes, (3, 3), padding = 'same')(stage5d)
    side5 = keras.layers.UpSampling2D((16, 16), name = 'side5')(side5)
    side6 = Conv2D(num_classes, (3, 3), padding = 'same')(stage6)
    side6 = keras.layers.UpSampling2D((16, 16), name = 'side6')(side6)

    out = Concatenate(axis = -1)([side1, side2, side3, side4, side5, side6])
    out = Conv2D(num_classes, (1, 1), padding = 'same', name = 'out')(out)

    # model = Model(inputs = [input], outputs = [side1, side2, side3, side4, side5, side6, out])
    model = Model(inputs = [input], outputs = [out])
    
    return model
    


# DnCNN Model
# ----------------------------------------------------------------------------------------------

def DnCNN(num_classes = num_classes, img_height = height, img_width = width, in_channels = in_channels):
    
    inpt = Input(shape=(img_height, img_width, in_channels))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(num_classes, (1, 1), activation='softmax')(x)
    # x = Conv2D(filters=6, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    # x = tf.keras.layers.Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    
    return model


if __name__ == '__main__':
    
    model = u2net()
    model.summary()
    