"""Multi-Layer Perceptron for Fashion MNIST Classification.

Team #name
"""
from __future__ import print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import to_categorical

from pnslib import utils
from pnslib import ml

# Load all the ten classes from Fashion MNIST
# complete label description is at
# https://github.com/zalandoresearch/fashion-mnist#labels
(train_x, train_y, test_x, test_y) = utils.fashion_mnist_load(
    data_type="full", flatten=True)

num_classes = 10

print("[MESSAGE] Dataset is loaded.")

# preprocessing for training and testing images
train_x = train_x.astype("float32") / 255.  # rescale image
mean_train_x = np.mean(train_x, axis=0)  # compute the mean across pixels
train_x -= mean_train_x  # remove the mean pixel value from image
test_x = test_x.astype("float32") / 255.
test_x -= mean_train_x

print("[MESSAGE] Dataset is preprocessed.")

# Use PCA to reduce the dimension of the dataset,
# so that the training will be less expensive
# perform PCA on training dataset
train_X, R, n_retained = ml.pca(train_x)

# perform PCA on testing dataset
test_X = ml.pca_fit(test_x, R, n_retained)

print("[MESSAGE] PCA is complete.")

# converting the input class labels to categorical labels for training
train_Y = to_categorical(train_y, num_classes=num_classes)
test_Y = to_categorical(test_y, num_classes=num_classes)

print("[MESSAGE] Converted labels to categorical labels.")

# define a model
# >>>>> PUT YOUR CODE HERE <<<<<
input_dim = train_X.shape[1]
inputs = Input(shape=(input_dim,))
x1 = Dense(100, activation='relu')(inputs)
x2 = Dense(100, activation='relu')(x1)
output = Dense(10, activation='softmax')(x2)
model = Model(inputs, output)

print("[MESSAGE] Model is defined.")

# print model summary
model.summary()

# compile the model aganist the categorical cross entropy loss
# and use SGD optimizer, you can try to use different
# optimizers if you want
# see https://keras.io/losses/
# >>>>> PUT YOUR CODE HERE <<<<<
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

print("[MESSAGE] Model is compiled.")

# train the model with fit function
# See https://keras.io/models/model/ for usage
# >>>>> PUT YOUR CODE HERE <<<<<
model.fit(x=train_X, y=train_Y, batch_size=64, epochs=10, validation_data=(test_X, test_Y))

print("[MESSAGE] Model is trained.")

# save the trained model
model.save("mlp-fashion-mnist-trained.hdf5")

print("[MESSAGE] Model is saved.")

# visualize the ground truth and prediction
# take first 10 examples in the testing dataset
test_X_vis = test_X[:10]  # fetch first 10 samples
ground_truths = test_y[:10]  # fetch first 10 ground truth prediction
# predict with the model
preds = np.argmax(model.predict(test_X_vis), axis=1).astype(np.int)

labels = ["Tshirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
          "Shirt", "Sneaker", "Bag", "Ankle Boot"]

plt.figure()
for i in range(2):
    for j in range(5):
        plt.subplot(2, 5, i * 5 + j + 1)
        plt.imshow(test_x[i * 5 + j].reshape(28, 28), cmap="gray")
        plt.title("Ground Truth: %s, \n Prediction %s" %
                  (labels[ground_truths[i * 5 + j]],
                   labels[preds[i * 5 + j]]))
plt.show()
