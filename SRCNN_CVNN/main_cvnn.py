"""
Author: Rayyan Abdalla
Email:rsabdall@asu.edu 

Image Super-resolution example using CVNN
    Python re-implementation of SRCNN model using CVNN  repository:https://github.com/NEGU93/cvnn
    Cited from SCRNN real-valued implementation: https://github.com/MarkPrecursor/SRCNN-keras

Image dataset is T91:  https://www.kaggle.com/datasets/ll01dm/t91-image-dataset
    real-valued data is transformed to complex domain using Hilbert transform

CVNN Library obtained from repository:
    https://github.com/NEGU93/cvnn

"""



from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization


# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import prepare_data as pd
import numpy
import math
import  numpy as np
import cv2

import cvnn.layers as complex_layers   # Ouw layers!
from cvnn.layers import complex_input

import matplotlib.pyplot as plt
from scipy.signal import hilbert


data, label = pd.read_training_data("./train.h5")
val_data, val_label = pd.read_training_data("./test.h5")


data_analytic = np.zeros(data.shape, dtype='complex64')

for cnt in range(data.shape[0]):
    # for chn in range(data.shape[3]):
    #     data_analytic[cnt,:,:,chn] = hilbert(data[cnt,:,:,chn])
    data_analytic[cnt,:,:,:] = hilbert(data[cnt,:,:,:])



val_data_analytic = np.zeros(val_data.shape, dtype='complex64')

for cnt in range(val_data.shape[0]):
    # for chn in range(val_data.shape[3]):
        # val_data_analytic[cnt,:,:,chn] = hilbert(val_data[cnt,:,:,chn])
        val_data_analytic[cnt,:,:,:] = hilbert(val_data[cnt,:,:,:])
        
#---------------------------------------------------------------------------------------
#%%
def psnr(target, ref):
    # assume RGB image
    target_data = numpy.array(target, dtype=float)
    ref_data = numpy.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(numpy.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)


def model():
    # lrelu = LeakyReLU(alpha=0.1)
    SRCNN = Sequential()
    SRCNN.add(Conv2D(128, (9,9),kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(64, (3,3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    # SRCNN.add(Conv2D(32, 1, kernel_initializer='glorot_uniform',
    #                  activation='relu', padding='same', use_bias=True))
    # SRCNN.add(BatchNormalization())
    SRCNN.add(Conv2D(1, (5,5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    adam = Adam(lr=0.003)
    # adam = SGD(learning_rate=0.01)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN



def complex_model():
    SRCNN_complex = Sequential()
    
    SRCNN_complex.add(complex_input(shape=(None,None,1)))
    
    SRCNN_complex.add(complex_layers.ComplexConv2D(64, (9,9),kernel_initializer='glorot_uniform',
                     activation='cart_relu', padding='valid', use_bias=True, input_shape=(None, None, 1), dtype='complex64'))
    
    SRCNN_complex.add(complex_layers.ComplexConv2D(32, (1,1), kernel_initializer='glorot_uniform',
                     activation='cart_relu', padding='same', use_bias=True,dtype='complex64'))
    
    SRCNN_complex.add(complex_layers.ComplexConv2D(1, (5,5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True,dtype='float32')) # Either tensorflow ' relu' or 'cart_relu' will work
    
    adam = Adam(lr=0.003)
    # adam = SGD(learning_rate=0.01)
    SRCNN_complex.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])

    return SRCNN_complex
    
    
    
    
    
    


def predict_model():
    # lrelu = LeakyReLU(alpha=0.1)
    SRCNN = Sequential()
    SRCNN.add(Conv2D(256, 9, kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(128, 3, kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    # SRCNN.add(BatchNormalization())
    SRCNN.add(Conv2D(1, 5, kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    adam =  Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN


def train():
    # srcnn_model = model()
    srcnn_model = complex_model()
    print(srcnn_model.summary())
    data = data_analytic; 
    val_data = val_data_analytic
  
    
    # data, label = pd.read_training_data("./train.h5")
    # val_data, val_label = pd.read_training_data("./test.h5")
    
    # breakpoint()
    checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                  save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]
    # breakpoint()

    history = srcnn_model.fit(data, label, batch_size=128, validation_data=(val_data, val_label),
                     callbacks=callbacks_list,shuffle=True, epochs=50, verbose=1)
    
    # srcnn_model.load_weights("m_model_adam.h5")
    
    
    
    #__________________________________________________________________________________________________
    
  
    
    IMG_NAME = ".//Test//Set14//baboon.bmp"
    #IMG_NAME = "/home/mark/Engineer/SR/data/Set14/flowers.bmp"
    INPUT_NAME = "input2.jpg"
    OUTPUT_NAME = "pre2.jpg"

    import cv2
    img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    shape = img.shape
    Y_img = cv2.resize(img[:, :, 0], (shape[1] // 2, shape[0] // 2), cv2.INTER_CUBIC)
    Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    img[:, :, 0] = Y_img
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(INPUT_NAME, img)

    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img.astype(float) / 255.
    pre = srcnn_model.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(numpy.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
    # img[9: -9, 9: -9, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(OUTPUT_NAME, img)

    # psnr calculation:
    im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    
    # im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[9: -9, 9: -9, 0]
    # im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
    # im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[9: -9, 9: -9, 0]
    # im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
    # im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[9: -9, 9: -9, 0]

    print ("bicubic:")
    print (cv2.PSNR(im1, im2))
    print ("SRCNN:")
    print (cv2.PSNR(im1, im3))
    
    cv2.imshow("Original",im1)
    cv2.imshow("bicubic",im2)
    cv2.imshow("SRCNN",im3)
    
    
    im4 = cv2.resize(im1, (im1.shape[1] // 2, im1.shape[0] // 2))
    
    # im4 = cv2.resize(im4, (im1.shape[1], im1.shape[0]))
    
    cv2.imshow("Original",im1)
    cv2.imshow("LR",im4)
    cv2.imshow("bicubic",im2)
    cv2.imshow("SRCNN",im3)
    cv2.waitKey(0)
    return history



# def predict():
#     srcnn_model = predict_model()
#     #srcnn_model.load_weights("3051crop_weight_200.h5")
#     IMG_NAME = ".//Test//Set14//flowers.bmp"
#     #IMG_NAME = ".//Test//Set14//flowers.bmp"
#     INPUT_NAME = "input2.jpg"
#     OUTPUT_NAME = "pre2.jpg"

#     import cv2
#     img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#     shape = img.shape
#     Y_img = cv2.resize(img[:, :, 0], (shape[1] // 2, shape[0] // 2), cv2.INTER_CUBIC)
#     Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
#     img[:, :, 0] = Y_img
#     img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
#     cv2.imwrite(INPUT_NAME, img)

#     Y = numpy.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
#     Y[0, :, :, 0] = Y_img.astype(float) / 255.
#     pre = srcnn_model.predict(Y, batch_size=1) * 255.
#     pre[pre[:] > 255] = 255
#     pre[pre[:] < 0] = 0
#     pre = pre.astype(numpy.uint8)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#     img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
#     img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
#     cv2.imwrite(OUTPUT_NAME, img)

#     # psnr calculation:
#     im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
#     im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
#     im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
#     im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
#     im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
#     im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]

#     print ("bicubic:")
#     print (cv2.PSNR(im1, im2))
#     print ("SRCNN:")
#     print (cv2.PSNR(im1, im3))


if __name__ == "__main__":
    hist = train()
   
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    
    
    #predict()
