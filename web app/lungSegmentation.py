
# Import Libraries and setup
from flask import Flask, request, g, redirect, url_for, flash, render_template, make_response
import requests
import os
import datetime
#import random
from pathlib import Path
import shutil
import numpy as np
from random import randint

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from keras.utils import normalize
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


MODEL_NAMES = ["lungSegmentation"]
print(MODEL_NAMES)


def test_lung(model, imagePath):
    # Resizing images, if needed
    image_name = os.path.split(imagePath)[1]
    image_name1 = os.path.splitext(image_name)[0]+'.png'
    print(image_name)
    SIZE_X = 352
    SIZE_Y = 352
    n_classes = 3  # Number of classes for segmentation
    #count = 0
    outfile1 = "D:\\EECS6895RealEngineeringProject\\webApp\\webApp1\\static\\lungSegmentation\\origin\\"
    outfile2 = "D:\\EECS6895RealEngineeringProject\\webApp\\webApp1\\static\\lungSegmentation\\mask\\"
    train_images = []
    img = cv2.imread(imagePath,0)
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    name1 = outfile1 + image_name 
    cv2.imwrite(name1, img)
    train_images.append(img)
    train_images = np.expand_dims(train_images, axis=3)
    train_images = normalize(train_images, axis=1)

    for test_img_number in range(len(train_images)):
        test_img = train_images[test_img_number]
        # ground_truth=y_test[test_img_number]
        test_img_norm = test_img[:, :, 0][:, :, None]
        test_img_input = np.expand_dims(test_img_norm, 0)
        prediction = (model.predict(test_img_input))
        predicted_img = np.argmax(prediction, axis=3)[0, :, :]

        #plt.figure(figsize=(20,8),dpi=80)
       # print("pred_img: ", pred.shape)
        # im = Image.fromarray( (pred * 255).astype(np.uint8))
        # im.save("pred.png")
        # draw_img[predicted_img] = [0,0,255]

        #plt.imshow(predicted_img)  # , cmap='jet')
        #plt.show()
        # cv2.imshow('GrayImage', predicted_img)
        im = Image.fromarray((predicted_img * 127).astype(np.uint8))
        im = im.convert('L')  
        #name1 = outfile1 + str(count) + '.png'
        name2 = outfile2 + image_name1
        im.save(name2)
        print("name1 is ", name1)
        print("name2 is ", name2)
        #print(IndexMap(name1 ,name2, SIZE_Y))
        return IndexMap(name1 ,name2, SIZE_Y)


def IndexMap(Origin_path, mask_path, size):
    folder1 = "D:\\EECS6895RealEngineeringProject\\webApp\\webApp1\\static\\lungSegmentation\\index\\"
    folder2 = "D:\\EECS6895RealEngineeringProject\\webApp\\webApp1\\static\\lungSegmentation\\final\\"
    img = Image.open(mask_path)
    print("mask_path: ", mask_path)
    name = os.path.split(mask_path)[1]
    name = os.path.splitext(name)[0] + '.png'
    print("name is:", name)
    filePath = folder1 + name  # Index graph

    finalImgPath =  os.path.join(app.config['UPLOAD_FOLDER1'], name)
    i = [0, 0, 0, 111, 177, 75, 242, 254, 139, 14, 142, 60, 18, 29, 229, 204, 130, 244, 90, 242, 223, 37, 149, 47, 58,
         14, 153, 240, 68, 195, 59, 163, 93, 164, 59, 163, 182, 212, 86, 229, 116, 12, 121, 223, 204, 148, 181, 83, 74,
         236, 250, 176, 241, 209, 64, 35, 43, 5, 54, 177, 123, 133, 61, 146, 82, 209, 126, 174, 118, 230, 201, 235, 204,
         136, 184, 166, 148, 185, 196, 235, 196, 61, 154, 236, 220, 178, 86, 207, 124, 88, 76, 189, 108, 207, 210, 41,
         57, 124, 107, 102, 132, 104, 186, 31, 141, 107, 126, 103, 6, 49, 215, 167, 161, 5, 201, 206, 244, 239, 70, 74,
         233, 55, 3, 37, 225, 28, 60, 183, 192, 59, 34, 84, 51, 116, 17, 188, 107, 236, 193, 11, 89, 60, 119, 216, 57,
         148, 123, 117, 99, 61, 230, 10, 220, 51, 248, 199, 47, 178, 105, 253, 31, 82, 209, 73, 144, 194, 39, 36, 118,
         48, 99, 164, 46, 80, 208, 197, 7, 229, 231, 91, 190, 102, 18, 169, 186, 184, 76, 48, 14, 59, 133, 39, 121, 74,
         68, 182, 144, 30, 209, 3, 49, 89, 102, 148, 119, 186, 208, 59, 128, 196, 141, 68, 149, 161, 43, 137, 126, 233,
         164, 244, 114, 33, 109, 93, 161, 217, 127, 131, 219, 27, 164, 87, 35, 147, 244, 97, 124, 8, 29, 65, 248, 55,
         133, 23, 160, 51, 105, 51, 12, 149, 1, 45, 205, 20, 12, 145, 89, 3, 2, 175, 240, 47, 23, 182, 160, 205, 147,
         235, 109, 148, 211, 120, 93, 73, 15, 158, 35, 90, 47, 185, 197, 92, 71, 167, 200, 255, 252, 217, 197, 112, 36,
         23, 95, 43, 19, 84, 110, 149, 70, 124, 90, 138, 17, 107, 144, 156, 166, 40, 122, 26, 43, 110, 160, 126, 138,
         149, 173, 203, 100, 116, 151, 10, 112, 153, 203, 154, 115, 183, 31, 197, 169, 75, 91, 26, 223, 229, 211, 101,
         240, 67, 58, 151, 230, 113, 124, 173, 95, 57, 33, 150, 14, 178, 5, 197, 110, 154, 37, 27, 112, 241, 109, 7,
         157, 145, 81, 176, 235, 30, 185, 216, 149, 179, 121, 180, 67, 103, 254, 79, 149, 64, 66, 219, 72, 121, 25, 30,
         59, 21, 24, 61, 125, 24, 163, 14, 139, 0, 204, 87, 46, 106, 248, 91, 172, 220, 59, 213, 155, 135, 101, 200, 20,
         106, 94, 24, 193, 145, 66, 38, 34, 74, 102, 15, 146, 191, 99, 164, 206, 235, 37, 172, 240, 222, 24, 186, 238,
         95, 26, 12, 66, 87, 140, 95, 216, 50, 160, 85, 194, 82, 21, 45, 247, 11, 200, 72, 61, 68, 160, 176, 218, 59,
         240, 66, 130, 224, 133, 103, 129, 83, 250, 132, 206, 16, 13, 154, 6, 147, 115, 196, 195, 5, 205, 218, 246, 109,
         137, 58, 2, 17, 242, 155, 159, 231, 109, 71, 122, 30, 174, 219, 20, 210, 148, 65, 20, 169, 238, 143, 46, 20,
         125, 10, 134, 84, 6, 87, 160, 85, 227, 223, 67, 42, 9, 225, 68, 232, 60, 53, 241, 225, 49, 57, 140, 8, 206,
         217, 73, 128, 93, 120, 117, 163, 65, 16, 204, 185, 64, 158, 109, 32, 222, 223, 86, 148, 249, 72, 192, 245, 231,
         36, 28, 108, 180, 48, 194, 199, 214, 49, 196, 201, 127, 1, 44, 220, 42, 123, 199, 41, 81, 49, 196, 251, 191,
         103, 51, 180, 58, 112, 60, 166, 214, 152, 71, 10, 34, 82, 240, 102, 247, 75, 204, 156, 89, 28, 76, 106, 24, 24,
         153, 181, 134, 204, 10, 103, 175, 146, 175, 151, 103, 186, 105, 168, 212, 38, 49, 30, 152, 63, 82, 225, 240,
         99, 192, 115, 126, 111, 16, 41, 141, 130, 28, 91, 53, 147, 207, 3, 11, 2, 202, 89, 223, 122, 217, 216, 112, 96,
         13, 250, 6, 204, 101, 26, 96, 45, 242, 177, 2, 190, 43, 23, 147, 228, 106, 47, 3, 117, 5, 106, 117, 110, 184,
         167, 246, 74, 13, 51, 95, 80, 156, 224, 194, 14, 120, 238, 224, 239, 92, 231, 219, 253, 45, 224, 17, 57, 50, 5,
         101, 238, 176, 221, 125, 154, 213, 56, 19, 139, 13, 39, 140, 142, 126, 22, 216, 83, 10, 81, 67, 220, 117, 66,
         211, 11, 231, 215, 97, 226, 122, 147, 136, 180, 194, 114, 176, 104, 78, 10, 62, 221, 142, 189, 216, 244, 151,
         55, 98, 106, 208, 220, 170, 177, 125, 119, 38, 193, 0, 255, 0, 19, 152, 170]
    # print (i)
    # print (len(i))
    img.putpalette(i)
    # print img
    # img.show()
    # if img.mode == "P":
    # img = img.convert('RGB')
    img.save(filePath)
    # 重叠原图和Index
    print("Final Image Path:",finalImgPath)
    origin_Img = cv2.imread(Origin_path)
    origin_Img = cv2.resize(origin_Img, (size, size))
    index_Img = cv2.imread(filePath)
    plt.figure(figsize=(20, 8), dpi=80)
    plt.imshow(origin_Img)
    plt.imshow(index_Img, alpha=0.6)
    plt.axis('off')
    plt.savefig(finalImgPath, bbox_inches='tight')
    return finalImgPath






UPLOAD_FOLDER1 = "static\\lungSegmentation\\final\\"
#UPLOAD_FOLDER2 = "static\\lungSegmentation\\origin\\"
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'PDF', 'PNG', 'JPG', 'JPEG','tif'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
app.config['UPLOAD_FOLDER1'] = UPLOAD_FOLDER1
filename='Image_No_Pred_MJRoBot.png'
image_name = filename

lunSeg_model = load_model('./model/Res-Unet-CrossEntropyCPULung.h5')



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/')
def hello_world():
	return render_template('LungSeg.html', ) #Home welcome page


@app.route("/LungSegquery", methods=["POST"])
def LungSegquery():

    if request.method == 'POST':
        # RECIBIR DATA DEL POST
        if 'file' not in request.files:
            return render_template('LungSeg.html',filename='no image')
        file = request.files['file']
        # image_data = file.read()
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('LungSeg.html', filename='no image')
        if file and allowed_file(file.filename):

            filename = str(file.filename) #0.jpg
            img_path = "static/lungSegmentation/origin/" + filename
            file.save(img_path) 
            image_name = filename
            path = test_lung(lunSeg_model, img_path)
            #output images to HTML
            return render_template('LungSeg.html', ori_lung_image= img_path ,Seg_lung_image=path, filename = filename)


        #else:
            #filename = str(file.filename)
            #image_name = filename
            #img_path = img_path = "static/brain_img/"+filename+".jpg"
            #return render_template('Brain.html', lung_image=img_path)





@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':

    app.run(host="0.0.0.0", port=5004, debug=False)
