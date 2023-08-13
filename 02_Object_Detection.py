#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import cv2


# In[2]:


df = pd.read_csv('bounding_boxes.csv')
df.head()


# In[3]:


import xml.etree.ElementTree as xet


# In[4]:


filename=df['filepath'][0]
filename


# In[6]:


def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('C:/Users/shiva/OneDrive/Desktop/anpr/XML',filename_image)
    return filepath_image


# In[7]:


getFilename(filename)


# In[8]:


image_path = list(df['filepath'].apply(getFilename))
image_path


# In[9]:


file_path = image_path[0]
file_path


# In[10]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# In[11]:


labels =  df.iloc[:,1:].values


# In[13]:


data = []
output = []
for ind in range(len(image_path)):
    image = image_path[ind]
    img_arr = cv2.imread(image)
    h,w,d = img_arr.shape
    #preprocessing
    load_image = load_img(image,target_size=(224,224))
    load_image_arr = img_to_array(load_image)
    norm_load_image_arr = load_image_arr/255.0 #normalisation
    #normalisation to labels
    xmin,xmax,ymin,ymax = labels[ind]
    nxmin,nxmax = xmin/w,xmax/w
    nymin,nymax = ymin/h,ymax/h
    label_norm = (nxmin,nxmax,nymin,nymax) #normalisation output
    #append
    data.append(norm_load_image_arr)
    output.append(label_norm)


# In[16]:


output


# In[18]:


x = np.array(data,dtype=np.float32)
y = np.array(output,dtype=np.float32)


# In[19]:


x.shape, y.shape


# In[21]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[24]:


from tensorflow.keras.applications import MobileNetV2, InceptionV3, InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
import tensorflow as tf


# In[25]:


inception_resnet = InceptionResNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Set the base model as non-trainable
inception_resnet.trainable = False

# Create the head of the model
headmodel = inception_resnet.output
headmodel = Flatten()(headmodel)
headmodel = Dense(500, activation="relu")(headmodel)
headmodel = Dense(250, activation="relu")(headmodel)
headmodel = Dense(4, activation='sigmoid')(headmodel)


# In[26]:


# Combine the base model and head model
model = Model(inputs=inception_resnet.input, outputs=headmodel)

# Compile the model (add loss and optimizer as per your requirements)
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

# Print a summary of the model architecture
model.summary()  


# In[27]:


from tensorflow.keras.callbacks import TensorBoard


# In[28]:


tfb = TensorBoard('object_detection')


# In[29]:


history = model.fit(x=x_train, y=y_train, batch_size=10, epochs=100, validation_data=(x_test, y_test), callbacks=[tfb])


# In[30]:


history = model.fit(x=x_train, y=y_train, batch_size=10, epochs=200, validation_data=(x_test, y_test), callbacks=[tfb],initial_epoch=101)


# In[31]:


model.save('C:/Users/shiva/OneDrive/Desktop/anpr/models/object_detection.h5')


# In[32]:


model.save('C:/Users/shiva/OneDrive/Desktop/anpr/models/object_detection.keras')




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




