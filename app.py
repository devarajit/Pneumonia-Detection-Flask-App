from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

import tensorflow as tf


import pydicom

from PIL import Image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template,send_file
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D,Flatten,Dense,Activation,BatchNormalization,Dropout,Reshape
from tensorflow.keras.utils import Sequence
from tensorflow.keras.backend import epsilon
import io
import base64
import h5py  

# Define a flask app
app = Flask(__name__)

ALPHA = 1.0 # Width hyper parameter for MobileNet (0.25, 0.5, 0.75, 1.0). Higher width means more accurate but slower
IMAGE_SIZE = 224 # Image sizes can vary (128, 160, 192, 224). MobileNetV2 can also take 96
ORIGINAL_IMAGE_SIZE = 1024
EPOCHS = 25 # Number of epochs. I got decent performance with just 5.
BATCH_SIZE = 8 # Depends on your GPU or CPU RAM.
PATIENCE = 50 # Patience for early stopping

def create_vggnet_conv2d_model(trainable=True):
  model = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False) # Load pre-trained mobilenet
  # Do not include classification (top) layer
  # to freeze layers, except the new top layer, of course, which will be added below
  for layer in model.layers:
      layer.trainable = trainable
  # Add new top layer which is a conv layer of the same size as the previous layer so that only 4 coords of BBox can be output
  x = model.layers[-1].output
  x = Conv2D(4, kernel_size=2, name="coords")(x)
  # In the line above kernel size should be 3 for img size 96, 4 for img size 128, 5 for img size 160 etc.
  x = Reshape((4,))(x) # These are the 4 predicted coordinates of one BBox
  return Model(inputs=model.input, outputs=x)

VGG_MODEL_PATH='models/vggnet_conv2d_model_july4.h5'
vgg_model = load_model(VGG_MODEL_PATH)
print(' VGG Model loaded. Start serving...')
Mobilenet_MODEL_PATH='models/vggnet_conv2d_model_july4.h5'
mobilenet_model = load_model(Mobilenet_MODEL_PATH)
print(' Mobilenet Model loaded. Start serving...')
maskcnn_MODEL_PATH='models/mask_rcnn_model.h5'
maskcnn_model = load_model(maskcnn_MODEL_PATH)
print(' MaskRCNN Model loaded. Start serving...')
def load_image(filename):
  print(filename)
  image = pydicom.dcmread(filename) # Read image
  print(image)
  image = image.pixel_array# Original image for display
  image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)) # Rescaled image to run the network
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image
  

def predict_model(image,model):
  # Pick a test image, run model, show image, and show predicted bounding box overlaid on the image
  feat_scaled = preprocess_input(np.array(image, dtype=np.float32))
  pred = model.predict(x=np.array([feat_scaled]))[0] # Predict the BBox
  predClass = int(np.argmax(pred))
  print(pred)
  print(predClass)
  x0 = int(pred[0] ) 
  y0 = int(pred[1] )
  x1 = int((pred[0] + pred[2]) )
  y1 = int((pred[1] + pred[3]) )
  print("predition Bounding Box co-ordinates are :",x0,y0,x1,y1)
  cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 1) # Show the BBox
  img = Image.fromarray(image.astype('uint8'))
  file_object = io.BytesIO()
  img.save(file_object, 'PNG')
  file_object.seek(0)
  return file_object

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        image = load_image(file_path)
    

        # Make prediction
        vgg_preds = predict_model(image, vgg_model)
        vgg_image = get_response_image(vgg_preds)
        mobile_preds = predict_model(image, mobilenet_model)
        mobile_image = get_response_image(mobile_preds)
        min_conf=0.8
        cnn_preds=maskcnn_model.detect(image)
        r=cnn_preds[0]
        if len(r['rois']) == 0: 
          message = 'Normal - No Pnuemonia'
        else: 
          num_instances = len(r['rois'])
          for i in range(num_instances): 
                if r['scores'][i] > min_conf: 
                  # x1, y1, width, height 
                  x0 = r['rois'][i][1]
                  y0 = r['rois'][i][0]
                  x1 = r['rois'][i][3] 
                  y1 = r['rois'][i][2] - y1 
                  print("predition Bounding Box co-ordinates are :",x0,y0,x1,y1)
                  cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 1) # Show the BBox
          img = Image.fromarray(image.astype('uint8'))        
          rcnn_image = io.BytesIO()
          img.save(rcnn_image, 'PNG')
          rcnn_image.seek(0)
        message = 'here is my message'
        response =  { 'Status' : 'Success', 
                        'message': message, 
                        'vgg': vgg_image,
                        'mobile': mobile_image,
                        'maskrcnn':rcnn_image}
        return response 
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)
    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
