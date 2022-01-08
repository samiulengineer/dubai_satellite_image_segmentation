import os
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, concatenate, Activation, MaxPool2D, Lambda
tf.config.experimental_run_functions_eagerly(True)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"


def unet(n_classes=6, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3):
    
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
 
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
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model
    
    
    
# Modified-U-NET Model

def mod_unet(n_classes=6, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3):
    
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
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
    
    
    
    #Expansive path 
    
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
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c13)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model






# U2-NET Model

class REBNCONV(Layer):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV,self).__init__()
        self.conv_s1 = Conv2D(out_ch, 3, padding="same")
        self.bn_s1 = BatchNormalization
        self.relu_s1 = ReLU()

    def call(self, inputs, **kwargs):
        xout = self.relu_s1(self.bn_s1()(self.conv_s1(inputs)))
        return xout

def _upsample_like(tensorA, tensorB):
    sB = K.int_shape(tensorB)
    def resize_like(tensor, sB): return tf.compat.v1.image.resize_bilinear(tensor, sB[1:3], align_corners=True)
    return Lambda(resize_like, arguments={'sB': sB})(tensorA)


class RSU7(Layer):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = MaxPool2D(2, strides=2)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = MaxPool2D(2,strides=2)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = MaxPool2D(2, strides=2)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = MaxPool2D(2, strides=2)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = MaxPool2D(2,strides=2)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def call(self, rsu7_input, **kwargs):

        hxin = self.rebnconvin(rsu7_input)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        # hx6d =  self.rebnconv6d(concatenate((hx7, hx6), 3))
        hx6d = self.rebnconv6d(concatenate([hx7, hx6], axis = 3))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d =  self.rebnconv5d(concatenate([hx6dup, hx5], axis = 3))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(concatenate([hx5dup, hx4], axis = 3))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(concatenate([hx4dup, hx3], axis = 3))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(concatenate([hx3dup, hx2], axis = 3))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(concatenate([hx2dup, hx1], axis = 3))
        # hx1dup = _upsample_like(hx1d, hxin)

        # return hx1dup + hxin
        return hx1d + hxin



    def summary(self):
        x = Input(shape=(256, 256, 3))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()



### RSU-6 ###
class RSU6(Layer):#UNet06DRES(Layer):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = MaxPool2D(2, strides=2)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = MaxPool2D(2, strides=2)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = MaxPool2D(2, strides=2)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = MaxPool2D(2, strides=2)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def call(self, rsu6_input, **kwargs):

        hxin = self.rebnconvin(rsu6_input)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(concatenate([hx6, hx5], axis = 3))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(concatenate([hx5dup, hx4], axis = 3))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(concatenate([hx4dup, hx3], axis = 3))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(concatenate([hx3dup, hx2], axis = 3))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(concatenate([hx2dup, hx1], axis = 3))
        hx1dup = _upsample_like(hx1d, hxin)

        return hx1dup + hxin

### RSU-5 ###
class RSU5(Layer):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = MaxPool2D(2, strides=2)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = MaxPool2D(2, strides=2)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = MaxPool2D(2, strides=2)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def call(self, rsu5_input, **kwargs):

        hxin = self.rebnconvin(rsu5_input)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(concatenate([hx5, hx4], axis = 3))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(concatenate([hx4dup, hx3], axis = 3))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(concatenate([hx3dup, hx2], axis = 3))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(concatenate([hx2dup, hx1], axis = 3))
        hx1dup = _upsample_like(hx1d, hxin)

        return hx1dup + hxin

### RSU-4 ###
class RSU4(Layer):#UNet04DRES(Layer):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = MaxPool2D(2, strides=2)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = MaxPool2D(2, strides=2)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def call(self, rsu4_input, **kwargs):

        hxin = self.rebnconvin(rsu4_input)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(concatenate([hx4, hx3], axis = 3))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(concatenate([hx3dup, hx2], axis = 3))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(concatenate([hx2dup, hx1], axis = 3))
        hx1dup = _upsample_like(hx1d, hxin)

        return hx1dup + hxin

### RSU-4F ###
class RSU4F(Layer):#UNet04FRES(Layer):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def call(self, rsu4f_input, **kwargs):

        hxin = self.rebnconvin(rsu4f_input)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(concatenate([hx4,hx3], axis = 3))
        hx2d = self.rebnconv2d(concatenate([hx3d,hx2], axis = 3))
        hx1d = self.rebnconv1d(concatenate([hx2d,hx1], axis = 3))

        hx1dup = _upsample_like(hx1d, hxin)

        return hx1dup + hxin


##### U^2-Net ####
class U2NET(Model):

    def __init__(self,in_ch=3, out_ch=6):
        super(U2NET,self).__init__()

        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = MaxPool2D(2,strides=2)

        self.stage2 = RSU6(64,32,128)
        self.pool23 = MaxPool2D(2,strides=2)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = MaxPool2D(2,strides=2)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = MaxPool2D(2,strides=2)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = MaxPool2D(2,strides=2)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = Conv2D(out_ch, 3, padding="same")
        self.side2 = Conv2D(out_ch, 3, padding="same")
        self.side3 = Conv2D(out_ch, 3, padding="same")
        self.side4 = Conv2D(out_ch, 3, padding="same")
        self.side5 = Conv2D(out_ch, 3, padding="same")
        self.side6 = Conv2D(out_ch, 3, padding="same")

        self.outconv = Conv2D(out_ch, 1)

    def call(self, u2net_input, **kwargs):

        #stage 1
        hx1 = self.stage1(u2net_input)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(concatenate([hx6up,hx5], axis = 3))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(concatenate([hx5dup,hx4], axis = 3))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(concatenate([hx4dup,hx3], axis = 3))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(concatenate([hx3dup,hx2], axis = 3))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(concatenate([hx2dup,hx1], axis = 3))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(concatenate([d1,d2,d3,d4,d5,d6], axis = 3))

        return Activation('softmax')(d0)
        # return Activation('softmax')(d0), Activation('softmax')(d1), Activation('softmax')(d2), Activation('softmax')(d3), Activation('softmax')(d4), Activation('sigmoid')(d5), Activation('sigmoid')(d6)


    def summary(self):
        x = Input(shape=(256, 256, 3))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()


### U^2-Net small ###
class U2NETP(Model):

    def __init__(self,in_ch=3, out_ch=6):
        super(U2NETP,self).__init__()

        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = MaxPool2D(2, strides=2 )

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = MaxPool2D(2, strides=2)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = MaxPool2D(2, strides=2)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = MaxPool2D(2, strides=2)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = MaxPool2D(2, strides=2)

        self.stage6 = RSU4F(64, 16, 64)

        # decoder
        self.stage5d = RSU4F(128, 16 ,64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = Conv2D(out_ch, 3, padding="same")
        self.side2 = Conv2D(out_ch, 3, padding="same")
        self.side3 = Conv2D(out_ch, 3, padding="same")
        self.side4 = Conv2D(out_ch, 3, padding="same")
        self.side5 = Conv2D(out_ch, 3, padding="same")
        self.side6 = Conv2D(out_ch, 3, padding="same")

        self.outconv = Conv2D(out_ch, 1)

    def call(self, u2netp_input, **kwargs):

        #stage 1
        hx1 = self.stage1(u2netp_input)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #decoder
        hx5d = self.stage5d(concatenate([hx6up,hx5], axis = 3))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(concatenate([hx5dup,hx4], axis = 3))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(concatenate([hx4dup,hx3], axis = 3))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(concatenate([hx3dup,hx2], axis = 3))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(concatenate([hx2dup,hx1], axis = 3))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(concatenate([d1, d2, d3, d4, d5, d6], axis = 3))

        return Activation('softmax')(d0)

        # return Activation('softmax')(d0), Activation('softmax')(d1), Activation('softmax')(d2), Activation('softmax')(d3), Activation('softmax')(d4), Activation('softmax')(d5), Activation('softmax')(d6)

    def summary(self):
        x = Input(shape=(256, 256, 3))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()
    
    


# DnCNN Model

def DnCNN():
    
    inpt = Input(shape=(256,256,3))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(6, (1, 1), activation='softmax')(x)
    # x = Conv2D(filters=6, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    # x = tf.keras.layers.Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    
    return model


if __name__ == '__main__':
    
    model = multi_unet_model()
    model.summary()
    