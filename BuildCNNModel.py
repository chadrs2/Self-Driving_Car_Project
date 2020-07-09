# -*- coding: utf-8 -*-
"""
Build My CNN (Convolutional Neural Network) Model based on the NVIDIA model

Designed and Run in Google Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HPC6jtdcV8qWVOCRJKxh4Q3YDU_40w-b
"""

!git clone https://github.com/chadrs2/myNewTrack.git

!ls myNewTrack

!pip3 install imgaug

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random

# Load CSV file into Data Object
datadir = 'myNewTrack'
columns = ['center','steering','throttle']
data = pd.read_csv(os.path.join(datadir, 'data_img.csv'), names = columns)
pd.set_option('display.max_colwidth',None)
data.head()

# Remove image paths
def path_leaf(path):
  head,tail = ntpath.split(path)
  return tail
data['center'] = data['center'].apply(path_leaf)
data.head()

# Make sure steering data is unbiased
num_bins = 25
samples_per_bin = 650
hist,bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1]+bins[1:]) * 0.5 #center bins
plt.bar(center, hist, width=0.25)
plt.plot((np.min(data['steering']),np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.show()

# Load image paths and steering values of servo motor
def load_img_steering(datadir1,datadir2,data):
  img_path = []
  steering = []
  for i in range(len(data)): #Each row of data object
    indexed_data = data.iloc[i]
    center = indexed_data[0]
    if i < 1098:
      img_path.append(os.path.join(datadir1,center.strip()))
      steering.append(float(indexed_data[1]))
    elif (i < 1182 or i > 1204): #else:
      img_path.append(os.path.join(datadir2,center.strip()))
      steering.append(float(indexed_data[1]))
  img_paths = np.asarray(img_path)
  steerings = np.asarray(steering)
  return img_paths,steerings
image_paths, steerings = load_img_steering(datadir + '/IMG',datadir + '/IMG2',data)
print(image_paths)
print(steerings)

# Split data randomly with 80% to be used as training data and 20% as validation data
X_train, X_valid, y_train, y_valid = train_test_split(image_paths,steerings,test_size=0.2,random_state=6)
#print(X_train)
#print(y_train)
print('Training Sample: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))
# Make sure data split kept steering unbiased
fig,axs = plt.subplots(1,2,figsize=(12,4))
axs[0].hist(y_train,bins=num_bins,width=0.25,color='blue')
axs[0].set_title('Training Set')
axs[1].hist(y_valid,bins=num_bins,width=0.25,color='red')
axs[1].set_title('Validation Set')

# Data augmentation techniques
# Zoom image randomly by 30%
def zoom(img):
  zoom = iaa.Affine(scale=(1,1.3))
  img = zoom.augment_image(img)
  return img
# Visualize zoom function
image = image_paths[random.randint(0,1500)]
original_image = mpimg.imread(image)
zoomed_image = zoom(original_image)
fig,axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(zoomed_image)
axs[1].set_title('Zoomed Image')

# Pan image randomly +-10%
def pan(img):
  pan = iaa.Affine(translate_percent={"x": (-0.1,0.1), "y": (-0.1, 0.1)})
  img = pan.augment_image(img)
  return img
# Visualize pan function
image = image_paths[random.randint(0,1500)]
original_image = mpimg.imread(image)
panned_image = pan(original_image)
fig,axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(panned_image)
axs[1].set_title('Panned Image')

# Randomly alter image brightness
def img_random_brightness(img):
  brightness = iaa.Multiply((0.2, 1.2))
  img = brightness.augment_image(img)
  return img
# Visualize brightness alteration
image = image_paths[random.randint(0,1500)]
original_image = mpimg.imread(image)
rand_image_brightness = img_random_brightness(original_image)
fig,axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(rand_image_brightness)
axs[1].set_title('Random Image Brightness')

# Randomly flip image horizontally and flip steering angle as well to match flipped image
def img_random_flip(img,steering_angle):
  img=cv2.flip(img,1) #1 = horizontal flip
  steering_angle = 12.75*2-steering_angle #13 means straight forward
  return img, steering_angle
# Visualize image & steering flip
random_idx = random.randint(0,1500)
image = image_paths[random_idx]
steering_angle = steerings[random_idx]
original_image = mpimg.imread(image)
rand_image_flip, flipped_steering_angle = img_random_flip(original_image, steering_angle)
fig,axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original Image - ' + 'Steering Angle ' + str(steering_angle))
axs[1].imshow(rand_image_flip)
axs[1].set_title('Random Image Flip - ' + 'Steering Angle ' + str(flipped_steering_angle))

# Randomly combine the assortment of image augmentation techniques on any one image
def random_augment(image,steering_angle):
  img = mpimg.imread(image)
  if np.random.rand() < 0.5: #occurs about 50% of the time
    img = pan(img)
  if np.random.rand() < 0.5:
    img = zoom(img)
  if np.random.rand() < 0.5:
    img = img_random_brightness(img)
  if np.random.rand() < 0.5:
    img,steering_angle = img_random_flip(img,steering_angle)
  return img, steering_angle
# Visualize random_augment function
numcols = 2
numrows = 10
fig,axs = plt.subplots(numrows,numcols,figsize=(15,50))
fig.tight_layout
for ii in range(numrows):
  randnum = random.randint(0,len(image_paths)-1)
  random_image = image_paths[randnum]
  random_steering = steerings[randnum]
  original_image = mpimg.imread(random_image)
  augmented_image, aug_steering = random_augment(random_image, random_steering)
  axs[ii][0].imshow(original_image)
  axs[ii][0].set_title("Original Image")
  axs[ii][1].imshow(augmented_image)
  axs[ii][1].set_title("Augmented Image")

# Preprocess image to match dimensions and color scheme used in the NVIDIA model
def img_preprocess(img):
  #img = mpimg.imread(img_path)
  img = img[20:240, :, :]
  #YUV is important when using NVIDIA Model
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) #Y=luminescence; U,V=chrominance
  img = cv2.GaussianBlur(img, (3,3), 0) #helps get rid of noise
  img = cv2.resize(img, (200, 66)) #matches size used in NVIDIA model architecture
  img = img/255 #normalize image
  return img
# Visualize preprocessed image
random_idx = random.randint(0,1500)
image = image_paths[random_idx]
orig_image = mpimg.imread(image)
preprocessed_image = img_preprocess(orig_image)
fig,axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()
axs[0].imshow(orig_image)
axs[0].set_title('Original Image')
axs[1].imshow(preprocessed_image)
axs[1].set_title('Preprocessed Image')
print(orig_image.shape)

# Randomly choose batch_size number of images from image_paths
def batch_generator(image_paths, steering_ang, batch_size, istraining):
  while(True):
    batch_img = []
    batch_steering = []
    for ii in range(batch_size):
      random_index = random.randint(0, len(image_paths)-1)
      if istraining:
        img, steering = random_augment(image_paths[random_index], steering_ang[random_index])
      else:
        img = mpimg.imread(image_paths[random_index])
        steering = steering_ang[random_index]
      img = img_preprocess(img)
      batch_img.append(img)
      batch_steering.append(steering)
    yield (np.asarray(batch_img), np.asarray(batch_steering))
# Test batch_generator function
#x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
#x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))
#fig,axs = plt.subplots(1,2,figsize=(15,10))
#fig.tight_layout()
#axs[0].imshow(x_train_gen[0])
#axs[0].set_title('Training Image')
#axs[1].imshow(x_valid_gen[0])
#axs[1].set_title('Validation Image')

# The buidling and layers inside my NVIDIA model for machine learning
def nvidia_model():
  model = Sequential()
  model.add(Conv2D(24, (5, 5), strides=(2,2), input_shape=(66,200,3), activation='elu'))
  model.add(Conv2D(36, (5,5), strides=(2,2), activation='elu'))
  model.add(Conv2D(48, (5,5), strides=(2,2), activation='elu'))
  model.add(Conv2D(64, (3,3), activation='elu'))
  model.add(Conv2D(64, (3,3), activation='elu'))

  model.add(Flatten())

  model.add(Dense(100, activation='elu'))

  model.add(Dense(50, activation='elu'))

  model.add(Dense(10, activation='elu'))
  #model.add(Dropout(0.5))

  model.add(Dense(1))

  model.compile(loss='mse', optimizer=Adam(lr=0.0001))
  return model
model = nvidia_model()
print(model.summary())

# Train & validate trained machine learning model on data
h = model.fit(batch_generator(X_train, y_train, 300, 1),
                        steps_per_epoch=300,
                        epochs=15,
                        validation_data = batch_generator(X_valid, y_valid, 150, 0),
                        validation_steps=200,
                        verbose=1,
                        shuffle=1)
# Plot resultant loss of model
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['Training Data', 'Validation Data'])
plt.title('Loss')
plt.xlabel('Epoch')

model.save('model.h5')

# Only used in Google Colab
#from google.colab import files
#files.download('model.h5')

# Convert model to a TensorFlow-Lite model to use in Raspberry Pi
from tensorflow import lite
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("model.tflite","wb").write(tflite_model)

#files.download('model.tflite')
