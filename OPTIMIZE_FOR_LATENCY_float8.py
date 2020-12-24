import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
import tensorflow.keras
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
import argparse
import io
import cv2




print("#set classes names")
classes_names = ['animals', 'other', 'person'] #you can change classes

print("#load model")
TF_LITE_MODEL_FILE_NAME = "quantization_OPTIMIZE_FOR_LATENCY_8bit_model_h5_to_tflite.tflite" #you can change model
interpreter = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE_NAME, num_threads=4)

print("#Check Input Tensor Shape")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("#Resize Tensor Shape")
interpreter.resize_tensor_input(input_details[0]['index'], (1, 299, 299, 3)) #you can change to your parameters
interpreter.resize_tensor_input(output_details[0]['index'], (1, 3)) #you can change to your parameters
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#load image and change size to 299*299
img_path = 'image.jpg'
img = image.load_img(img_path, target_size=(299, 299))

#convert image to array
new_img = image.img_to_array(img)
new_img /= 255
new_img = np.expand_dims(new_img, axis=0)

i = 1
while i <= 100:
    
    #predict class for image
    # input_details[0]['index'] = the index which accepts the input
    interpreter.set_tensor(input_details[0]['index'], new_img)
    # run the inference
    interpreter.invoke()   
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    #print predict classes
    classes = np.argmax(output_data, axis = 1)
    print("predict class number: ", classes, " ,is class name: ", classes_names[classes[0]], sep='')
    i += 1
