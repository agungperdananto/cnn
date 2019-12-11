import os, cv2, sys
# import keras
# import numpy as np
# from keras.models import load_model, model_from_json
import json
from flask import Flask, render_template, request, url_for, redirect, flash, session
from redis import Redis
from werkzeug import secure_filename

import keras
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.utils.vis_utils import model_to_dot

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# json_file = open('model/model-2.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# model.load_weights('model/model1-2.h5')

RESULT_FOLDER = os.path.join('static', 'result')
ALLOWED_EXTENSIONS = set([ 'pdf'])
app = Flask(__name__)
redis = Redis(host='redis', port=6379)

app.config['UPLOAD_FOLDER'] = RESULT_FOLDER
app.secret_key = "random string"

img_row = 28
img_col = 28
num_channel = 1
# model = None
# json_file = None
# loaded_model_json = None

def predict(data):
   json_file = open('model/model-2.json', 'r')
   loaded_model_json = json_file.read()
   json_file.close()
   model = model_from_json(loaded_model_json)
   model.load_weights('model/model1-2.h5')

   input_img = cv2.imread(data)
   input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
   input_img_resize=cv2.resize(input_img,(img_row,img_col))
   input_img_resize = input_img_resize.reshape(img_row*img_col)

   image = input_img_resize /255
   image = image.reshape(1,28,28,1)
   labels = ['Normal\n', 'Pneumonia\n']

   y_pred = model.predict(x=image)
   cls_pred = np.argmax(y_pred,axis=1)
   if y_pred[0,0] > y_pred[0,1]:
      confidence = y_pred[0,0]
   else:
      confidence = y_pred[0,1]
   keras.backend.clear_session()
   print("diagnosa: ", labels[cls_pred[0]], "confidence: ", confidence)
   return labels[cls_pred[0]], confidence

@app.route('/')
def home():
   # model = None
   # json_file = None
   # loaded_model_json = None
   full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'lung.png')
   print(full_filename)
   return render_template('home.html', result_image = full_filename)

# @app.route('/predict')
# def predict_data():
   
#    json_file = open('model/model-2.json', 'r')
#    loaded_model_json = json_file.read()
#    json_file.close()
#    model = model_from_json(loaded_model_json)
#    model.load_weights('model/model1-2.h5')

#    input_img = cv2.imread('static/result/NORMAL.jpeg')
#    input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
#    input_img_resize=cv2.resize(input_img,(img_row,img_col))
#    input_img_resize = input_img_resize.reshape(img_row*img_col)

#    image = input_img_resize /255
#    image = image.reshape(1,28,28,1)
#    labels = ['Normal\n', 'Pneumonia\n']

#    y_pred = model.predict(x=image)
#    cls_pred = np.argmax(y_pred,axis=1)
#    if y_pred[0,0] > y_pred[0,1]:
#       confidence = round(y_pred[0,0], 2)
#    else:
#       confidence = round(y_pred[0,1], 2)
#    keras.backend.clear_session()
#    print("diagnosa: ", labels[cls_pred[0]], "confidence: ", confidence)
#    return labels[cls_pred[0]]

@app.route('/result', methods = ['POST', 'GET'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      # filename = (secure_filename(f.filename))
      # f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      # full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      if f:
         filename = (secure_filename(f.filename))
         f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
         full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
         diagnosa, confidence = predict(full_filename)
         
      else:
         error = 'Pilih File Terlebih dahulu'
         flash(error)
         # return render_template('home.html')
         return redirect(url_for('home'))
      return render_template('result.html', result_image = full_filename, diagnosa=diagnosa, confidence=round(confidence *100,2))
      
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')

