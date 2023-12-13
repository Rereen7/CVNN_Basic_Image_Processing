# -*- coding: utf-8 -*-
"""
Author: Rayyan Abdalla
Email:rsabdall@asu.edu 

Image classification example using CVNN

Image dataset is CIFAR10, real-valued data is transformed to complex domain using Hilbert transform

CVNN Library obtained from repository:
    https://github.com/NEGU93/cvnn

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models
import cvnn.layers as complex_layers   # Ouw layers!
from cvnn.layers import complex_input
from cvnn.activations import analytic_relu

from scipy.signal import hilbert

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images.astype(dtype=np.float32) / 255.0, test_images.astype(dtype=np.float32) / 255.0


#%%
# Hilbert transform 

train_images_analytic = np.zeros(train_images.shape, dtype='complex64')

for cnt in range(train_images.shape[0]):
    # for chn in range(train_images.shape[3]):
        # train_images_analytic[cnt,:,:,chn] = hilbert(train_images[cnt,:,:,chn])
    train_images_analytic[cnt,:,:,:] = hilbert(train_images[cnt,:,:,:])


test_images_analytic = np.zeros(test_images.shape, dtype='complex64')

for cnt in range(test_images.shape[0]):
    # for chn in range(test_images.shape[3]):
    #     test_images_analytic[cnt,:,:,chn] = hilbert(test_images[cnt,:,:,chn])
    test_images_analytic[cnt,:,:,:] = hilbert(test_images[cnt,:,:,:])









#%%

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     # The CIFAR labels happen to be arrays,
#     # which is why you need the extra index
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()


model = models.Sequential()
model.add(complex_input(shape=(32,32,3)))
model.add(complex_layers.ComplexConv2D(32, (3, 3), activation='cart_relu', input_shape=(32, 32, 3), dtype=np.complex64))
model.add(complex_layers.ComplexMaxPooling2D((2, 2), dtype=np.complex64))
model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu', dtype=np.complex64)) # Either tensorflow ' relu' or 'cart_relu' will work
model.add(complex_layers.ComplexMaxPooling2D((2, 2), dtype=np.complex64))
model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu', dtype=np.complex64))

model.summary()

model.add(complex_layers.ComplexFlatten())
model.add(complex_layers.ComplexDense(32, activation='cart_relu', dtype=np.complex64))
model.add(complex_layers.ComplexDense(10, dtype=np.float32))

model.summary()



model.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images_analytic, train_labels, epochs=10, batch_size=50,validation_data=(test_images_analytic, test_labels))




plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images_analytic,  test_labels, verbose=1)