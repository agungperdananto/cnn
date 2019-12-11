import keras
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.utils.vis_utils import model_to_dot


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('data/data_train_mnist.csv')
test_df = pd.read_csv('data/data_test_mnist.csv')


train_df = np.array(train_df,dtype='float32')
test_df = np.array(test_df,dtype='float32')
x_train =  train_df[:,1:] /255
y_train =  train_df[:,0]

x_test =  test_df[:,1:] /255
y_test =  test_df[:,0]


# x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.2, random_state=12345)
x_validate = x_test
y_validate = y_test

im_rows = 28
im_cols = 28
batch_size = 512
im_shape = (im_rows, im_cols, 1)

x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_test = x_test.reshape(x_test.shape[0], *im_shape)
x_validate = x_validate.reshape(x_validate.shape[0], *im_shape)

print('x_train shape:{}'.format(x_train.shape))
print('x_test shape:{}'.format(x_test.shape))
print('x_validate shape:{}'.format(x_validate.shape))

name = 'model_1'
cnn_model_1 = Sequential([
    Conv2D(filters=32, kernel_size=1, activation='relu', input_shape = im_shape, name='Conv2D-1'),
    MaxPooling2D(pool_size=1, name='MaxPool-1'),
    Conv2D(filters=64, kernel_size=1, activation='relu', input_shape = im_shape, name='Conv2D-2'),
    MaxPooling2D(pool_size=2, name='MaxPool-2'),
    Dropout(rate=0.2, name='Dropout'),
    Flatten(name='Flatten'),
    Dense(128, activation='relu', name='Dense_1'),
    Dense(128, activation='relu', name='Dense_2'),
    Dense(2, activation='softmax', name='Output')
    
],name=name)

name = 'model_2'
cnn_model_2 = Sequential([
    Conv2D(filters=32, kernel_size=1, activation='relu', input_shape = im_shape, name='Conv2D-1'),
    AveragePooling2D(pool_size=1, name='avgPool-1'),
    Conv2D(filters=128, kernel_size=1, activation='relu', input_shape = im_shape, name='Conv2D-2'),
    AveragePooling2D(pool_size=2, name='AvgPool-2'),
    Dropout(rate=0.2, name='Dropout'),
    Flatten(name='Flatten'),
    Dense(128, activation='relu', name='Dense_1'),
    Dense(128, activation='relu', name='Dense_2'),
    Dense(2, activation='softmax', name='Output')
    
],name=name)

cnn_models = [cnn_model_1, cnn_model_2]
              #cnn_model_3]

history_dict = {}
for model in cnn_models:
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer = Adam(),
        metrics=['accuracy'])
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=10, verbose=True,
        validation_data=(x_validate, y_validate))
    history_dict[model.name] = history

score = cnn_model_1.evaluate(x_test, y_test, verbose=1)
print('Test Loss : {:.4f}'.format(score[0]))
print('Test Acc  : {:.4f}'.format(score[1]))


score = cnn_model_2.evaluate(x_test, y_test, verbose=1)
print('Test Loss : {:.4f}'.format(score[0]))
print('Test Acc  : {:.4f}'.format(score[1]))

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
# This is used for plotting the images.
img_shape = (img_size, img_size)

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
img_shape_full = (img_size, img_size, 1)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 2

labels = ['Normal\n', 'Pneumonia\n']

model1_json = cnn_model_1.to_json()
with open("model/model-2.json", "w") as json_file:
    json_file.write(model1_json)

cnn_model_1.save_weights("model/model1-2.h5")

print('finished.')