
# Import Libraries and setup
from flask import Flask, request, g, redirect, url_for, flash, render_template, make_response
import requests
import os
import datetime
#import random
from pathlib import Path
import shutil
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import cv2
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


MODEL_NAMES = ["CT","X-ray"]
AMOUNT_OF_MODELS = len(MODEL_NAMES)
print(MODEL_NAMES)
MODELS=[]

def load_models():
    for i in range(AMOUNT_OF_MODELS):
        load_single_model("./model/"+MODEL_NAMES[i]+".h5")
        print("Model " + str(MODELS[i]) + "loaded")

def load_single_model(path):
    model = load_model(path)
    MODELS.append(model)



# Functions
def test_Xray(model, imagePath, filename):
    img = cv2.imread(imagePath)
    img_out = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    img = np.array(img) / 255.0

    pred = model.predict(img)
    #pred_neg = int(round(pred[0][1]*100))
    #pred_pos = int(round(pred[0][0]*100))
    pred_covid = int(round(pred[0][0]*100))
    pred_normal = int (round(pred[0][1]*100))
    pred_pneumo = int (round(pred[0][2]*100))
    if np.argmax(pred, axis=1) == 0:
        prediction = 'COVID'
        prob = pred_covid
    elif np.argmax(pred, axis=1) == 1:
        prediction = 'NORMAL'
        prob = pred_normal
    else:
        prediction = 'PNEUMO'
        prob = pred_pneumo

    img_pred_name =  prediction+'_Prob_'+str(prob)+'_Name_'+filename+'.png'
    cv2.imwrite('static/xray_analisys/'+img_pred_name, img_out)
    cv2.imwrite('static/Image_Prediction.png', img_out)
    print
    return prediction, prob


def test_CT(model, imagePath, filename):
    img = cv2.imread(imagePath)
    img_out = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = np.expand_dims(img, axis=0)

    img = np.array(img) / 255.0

    pred = model.predict(img)
    #pred_neg = int(round(pred[0][1]*100))
    #pred_pos = int(round(pred[0][0]*100))
    pred_covid = int(round(pred[0][0]*100))
    pred_normal = int (round(pred[0][1]*100))
    #pred_pneumo = int (round(pred[0][2]*100))
    if np.argmax(pred, axis=1) == 0:
        prediction = 'COVID'
        prob = pred_covid
    elif np.argmax(pred, axis=1) == 1:
        prediction = 'NORMAL'
        prob = pred_normal


    img_pred_name =  prediction+'_Prob_'+str(prob)+'_Name_'+filename+'.png'
    cv2.imwrite('static/ct_analisys/'+img_pred_name, img_out)
    cv2.imwrite('static/Image_Prediction.png', img_out)
    print
    return prediction, prob



UPLOAD_FOLDER1 = os.path.join('static', 'xray_img')
UPLOAD_FOLDER2 = os.path.join('static', 'ct_img')
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'PDF', 'PNG', 'JPG', 'JPEG'])




app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
prediction=' '
confidence=0
filename='Image_No_Pred_MJRoBot.png'
image_name = filename
#ct_model = load_model('./model/CT.h5')
#xray_model = load_model('./model/X-ray.h5')
#load_models()
#app.config['UPLOAD_FOLDER1'] = UPLOAD_FOLDER1
#app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2
#def allowed_file(filename):
    #return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/')
def hello_world():
	return render_template('Welcome.html', ) #Home welcome page

#@app.route('/xray')
#def xray():
#    return render_template('Xray.html', ) #X-ray page
'''
@app.route("/Xrayquery", methods=["POST"])
def Xrayquery():
    if request.method == 'POST':
        # RECIBIR DATA DEL POST
        if 'file' not in request.files:
            return render_template('Xray.html', prediction='INCONCLUSIVE', confidence=0, filename='no image')
        file = request.files['file']
        # image_data = file.read()
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('Xray.html', prediction='INCONCLUSIVE', confidence=0, filename='no image')
        if file and allowed_file(file.filename):

            filename = str(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER1'], filename)
            file.save(img_path) 
            image_name = filename

            # detection covid
            try:
                prediction, prob = test_Xray(MODELS[0], img_path, filename)
                return render_template('Xray.html', prediction=prediction, confidence=prob, filename=image_name, xray_image=img_path)
            except:
                return render_template('Xray.html', prediction='INCONCLUSIVE', confidence=0, filename=image_name, xray_image=img_path)
        else:
            filename = str(file.filename)
            image_name = filename
            img_path = os.path.join(app.config['UPLOAD_FOLDER1'], filename)
            return render_template('Xray.html', name='FILE NOT ALLOWED', confidence=0, filename=image_name, xray_image=img_path)


'''
'''
@app.route('/ct')
def ct():
    return render_template('CT.html', ) #X-ray页面
@app.route("/CTquery", methods=["POST"])
def CTquery():
    if request.method == 'POST':
        # RECIBIR DATA DEL POST
        if 'file' not in request.files:
            return render_template('CT.html', prediction='INCONCLUSIVE', confidence=0, filename='no image')
        file = request.files['file']
        # image_data = file.read()
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('CT.html', prediction='INCONCLUSIVE', confidence=0, filename='no image')
        if file and allowed_file(file.filename):

            filename = str(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER2'], filename)
            file.save(img_path)
            image_name = filename

            # detection covid

            try:
                prediction, prob = test_CT(MODELS[1], img_path, filename)
                return render_template('CT.html', prediction=prediction, confidence=prob, filename=image_name, ct_image=img_path)
            except:
                return render_template('CT.html', prediction='INCONCLUSIVE', confidence=0, filename=image_name, ct_image=img_path)
        else:
            filename = str(file.filename)
            image_name = filename
            img_path = os.path.join(app.config['UPLOAD_FOLDER2'], filename)
            return render_template('CT.html', name='FILE NOT ALLOWED', confidence=0, filename=image_name, ct_image=img_path)
# No caching at all for API endpoints.
'''




@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':

    app.run("0.0.0.0", 5000, debug=False)
