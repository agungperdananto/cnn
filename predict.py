import os, cv2, sys
# import keras
# import numpy as np
# from keras.models import load_model, model_from_json
import json
from flask import Flask, render_template, request, url_for, redirect, flash, session
from werkzeug import secure_filename

import keras
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

json_file = open('model/model-2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model/model1-2.h5')

img_row = 28
img_col = 28
num_channel = 1

input_img = cv2.imread('static/result/NORMAL2-IM-1440-0001.jpeg')
input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
input_img_resize=cv2.resize(input_img,(img_row,img_col))
input_img_resize = input_img_resize.reshape(img_row*img_col)
image = input_img_resize /255
image = image.reshape(1,28,28,1)
labels = ['Normal\n', 'Pneumonia\n']

y_pred = model.predict(x=image)
cls_pred = np.argmax(y_pred,axis=1)
print(round(y_pred[0,0], 2))
print(round(y_pred[0,1], 2))