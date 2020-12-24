# Importing the os library
import os
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from PIL import Image, ImageOps

#set classes names
classes_names = ['animals', 'other', 'person']

print("#load model")
model = tensorflow.keras.models.load_model('mobilenetv2.h5')

#load image and change size to 299*299
img_path = 'image.jpg'
img = image.load_img(img_path, target_size=(299, 299))
 
#convert image to array
x = image.img_to_array(img)
x /= 255
x = np.expand_dims(x, axis=0)

i = 1
while i <= 100:
    #predict class for image
    prediction = model.predict(x)
    classes = np.argmax(prediction, axis = 1)
 
    #print class for image
    print("#predict class number: ", classes, " ,is class name: ", classes_names[classes[0]], sep='')

    #show time
    named_tuple = time.localtime()
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    print(time_string)
    i += 1