
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
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


MODEL_NAMES = ["X-ray"]
AMOUNT_OF_MODELS = len(MODEL_NAMES)
print(MODEL_NAMES)
MODELS=[]

#def load_models():
    #for i in range(AMOUNT_OF_MODELS):
        #load_single_model("./model/"+MODEL_NAMES[i]+".h5")
        #print("Model " + str(MODELS[i]) + "loaded")

#def load_single_model(path):
    #model = load_model(path)
    #MODELS.append(model)



# Functions
def test_Xray(model, imagePath, heatmap_path):
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
    #cv2.imwrite('static/xray_analisys/'+img_pred_name, img_out)
    cv2.imwrite('static/Image_Prediction.png', img_out)
    #print
    Heatmap(imagePath, model,heatmap_path)
    return prediction, prob

def Heatmap(imagePath,model,heatmap_path):
    IMAGE_PATH = imagePath
    LAYER_NAME = 'conv5_block3_out'
    CLASS_INDEX = 0
    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Load initial model
    model = model

    # Create a graph that outputs target convolution and output
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])

    # Get the score for target class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img]))
        loss = predictions[:, CLASS_INDEX]

    # Extract filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    # Average gradients spatially
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # Build a ponderated map of filters according to gradients importance
    cam = np.ones(output.shape[0:2], dtype=np.float32)

    for index, w in enumerate(weights):
        cam += w * output[:, :, index]

    # Heatmap visualization
    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)
    plt.imshow(output_image, cmap='rainbow')
    plt.savefig(heatmap_path,bbox_inches='tight')

ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'PDF', 'PNG', 'JPG', 'JPEG'])
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
prediction=' '
confidence=0
filename='Image_No_Pred_MJRoBot.png'
image_name = filename

xray_model = load_model('./model/X-ray.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/')
def hello_world():
	return render_template('Xray.html', ) #Home welcome page



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
            heatmapname = os.path.splitext(filename)[0] + '.jpeg'
            img_path = 'static/xray/xray_img/'+filename
            heatmap_path = 'static/xray/heatmap/'+heatmapname
            #img_path = os.path.join(app.config['UPLOAD_FOLDER1'], filename)
            #heatmap_path = os.path.join(app.config['UPLOAD_FOLDER2'], filename)
            file.save(img_path) 
            image_name = filename

            # detection covid
            try:
                prediction, prob = test_Xray(xray_model, img_path, filename, heatmap_path)
                return render_template('Xray.html', prediction=prediction, confidence=prob, filename=image_name, xray_image=img_path, heatmap=heatmap_path)
            except:
                return render_template('Xray.html', prediction='INCONCLUSIVE', confidence=0, filename=image_name, xray_image=img_path, heatmap=heatmap_path)
        else:
            filename = str(file.filename)
            image_name = filename
            img_path = os.path.join(app.config['UPLOAD_FOLDER1'], filename)
            return render_template('Xray.html', name='FILE NOT ALLOWED', confidence=0, filename=image_name, xray_image=img_path)






@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':

    app.run(host="0.0.0.0", port=5001, debug=False)
