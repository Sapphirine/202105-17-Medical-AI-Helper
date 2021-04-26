
# Import Libraries and setup
from flask import Flask, request, g, redirect, url_for, flash, render_template, make_response
import requests
import os
import datetime
#import random
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import cv2
from tensorflow.keras.models import load_model
import random
import pandas as pd
#import matplotlib.pyplot as plt
#plt.style.use("ggplot")
#%matplotlib inline
from tqdm import tqdm_notebook, tnrange
from glob import glob
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
#from skimage.color import rgb2gray
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras import backend as K
from PIL import Image
from frcnn import FRCNN

frcnn = FRCNN()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'PDF', 'PNG', 'JPG', 'JPEG','tif'])



app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
prediction=' '
confidence=0
filename='Image_No_Pred_MJRoBot.png'
image_name = filename

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/')
def hello_world():
	return render_template('Brain.html', ) #Home welcome page



@app.route("/Endosopic", methods=["POST"])
def Eadquery():

    if request.method == 'POST':
        # RECIBIR DATA DEL POST
        if 'file' not in request.files:
            return render_template('Brain.html',filename='no image')
        file = request.files['file']
        # image_data = file.read()
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('Brain.html', filename='no image')
        if file and allowed_file(file.filename):
            filename = str(file.filename)  # 0.jpg
            filename = os.path.splitext(filename)[0] + '.png'
            #print("filename:", filename)
            img_path = "static/brain/brain_img/" + filename
            file.save(img_path)  # Saved the image to CT_ IMG folder
            image = Image.open(img_path)
            image = image.convert("RGB")
            path = "static/brain/brainAna;isys/" + filename
            plt.savefig(image,path)
            return render_template('Brain.html', ori_image=img_path, res_image=path)


@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':

    app.run(host="0.0.0.0", port=5003, debug=False)
