from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import png, pydicom

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.wsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/trained_model.h5'

#Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64)) #target_size must agree with what the trained model expects!!

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

   
    preds = model.predict(img)
    return preds

def dicom2png(source_folder, output_folder):
    list_of_files = os.listdir(source_folder)
    for file in list_of_files:
        if not file.endswith('.dcm'):
            continue
        try:
            ds = pydicom.dcmread(os.path.join(source_folder,file))
            shape = ds.pixel_array.shape

            # Convert to float to avoid overflow or underflow losses.
            image_2d = ds.pixel_array.astype(float)

            # Rescaling grey scale between 0-255
            image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

            # Convert to uint
            image_2d_scaled = np.uint8(image_2d_scaled)

            # Write the PNG file
            with open(os.path.join(output_folder,file)+'.png' , 'wb') as png_file:
                w = png.Writer(shape[1], shape[0], greyscale=True)
                w.write(png_file, image_2d_scaled)
        except:
            print('Could not convert: ', file)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads or ./dcm_png
        basepath = os.path.dirname(__file__)
        input_folder = os.path.join(basepath,'dicom_png')
        output_folder = os.path.join(basepath,'uploads')
        str1 = 'Pneumonia'
        str2 = 'Normal'
        
        if f.filename.endswith('.dcm'):
            dicom_path = os.path.join(input_folder, secure_filename(f.filename))
            f.save(dicom_path)
            print('dicom file successfully saved')

            # convert dicom to png
            dicom2png(input_folder,output_folder)
            os.remove(dicom_path)
            list_of_output = os.listdir(output_folder)
            print(list_of_output)
            for file in list_of_output:
                if file.endswith('.png'):
                    preds = model_predict(file, model)
                    if preds == 1:
                        return str1
                    else:
                        return str2  
        else:
            file_path = os.path.join(output_folder, secure_filename(f.filename))
            f.save(file_path)
            preds = model_predict(file_path, model)
            os.remove(file_path) 
            if preds == 1:
                return str1
            else:
                return str2    
        

    return None

    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':
        app.run()
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
