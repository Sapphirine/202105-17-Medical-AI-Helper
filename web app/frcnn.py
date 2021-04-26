import colorsys
import copy
import math
import os
import pickle

import cv2
import keras
import numpy as np
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from PIL import Image, ImageDraw, ImageFont

import nets.frcnn as frcnn
from nets.frcnn_training import get_new_img_size
from utils.anchors import get_anchors
from utils.config import Config
from utils.utils import BBoxUtility


#--------------------------------------------#
#   Two parameters need to be modified to use the self trained model
#   model_ Path and classes_ Path needs to be modified!
#   If there is a shape mismatch
#   Be sure to pay attention to num during training_ CLASSES
#   model_ Path and classes_ Modification of path parameter
#--------------------------------------------#
class FRCNN(object):
    _defaults = {
        "model_path"    : '/home/gxt/study/faster-rcnn-keras-master/logs/Epoch100-Total_Loss0.8886-Val_Loss1.0822.h5',
        "classes_path"  : 'model_data/voc_classes.txt',
        "confidence"    : 0.5,
        "iou"           : 0.3
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   Initialize fast RCNN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.sess = K.get_session()
        self.config = Config()
        self.generate()
        self.bbox_util = BBoxUtility(classifier_nms=self.iou, top_k=self.config.num_RPN_predict_pre)

    #---------------------------------------------------#
    #   Get all categories
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   Load model
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        #-------------------------------#
        #   Calculate the total number of classes
        #-------------------------------#
        self.num_classes = len(self.class_names)+1

        #-------------------------------#
        #   Loading model and weight
        #-------------------------------#
        self.model_rpn, self.model_classifier = frcnn.get_predict_model(self.config, self.num_classes)
        self.model_rpn.load_weights(self.model_path, by_name=True)
        self.model_classifier.load_weights(self.model_path, by_name=True)
                
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Set different colors for the frame
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
    
    #---------------------------------------------------#
    #   Used to calculate the size of shared feature layers
    #---------------------------------------------------#
    def get_img_output_length(self, width, height):
        def get_output_length(input_length):
            filter_sizes = [7, 3, 1, 1]
            padding = [3,1,0,0]
            stride = 2
            for i in range(4):
                # input_length = (input_length - filter_size + stride) // stride
                input_length = (input_length + 2*padding[i]-filter_sizes[i]) // stride + 1
            return input_length
        return get_output_length(width), get_output_length(height) 
    
    #---------------------------------------------------#
    #   Detect images
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        old_width, old_height = image_shape[1], image_shape[0]
        old_image = copy.deepcopy(image)
    
        #---------------------------------------------------------#
        #   Reset the original image to the size of 600 short edges
        #---------------------------------------------------------#
        width, height = get_new_img_size(old_width, old_height)
        image = image.resize([width,height], Image.BICUBIC)
        photo = np.array(image,dtype = np.float64)

        #-----------------------------------------------------------#
        #   Image preprocessing and normalization.
        #-----------------------------------------------------------#
        photo = preprocess_input(np.expand_dims(photo,0))
        rpn_pred = self.model_rpn.predict(photo)

        #-----------------------------------------------------------#
        #   The prediction result of the suggestion box network is decoded
        #-----------------------------------------------------------#
        base_feature_width, base_feature_height = self.get_img_output_length(width, height)
        anchors = get_anchors([base_feature_width, base_feature_height], width, height)
        rpn_results = self.bbox_util.detection_out_rpn(rpn_pred, anchors)
        
        #-------------------------------------------------------------#
        #   After obtaining the suggestion box and the shared feature layer, they are passed into the classifier for prediction
        #-------------------------------------------------------------#
        base_layer = rpn_pred[2]
        proposal_box = np.array(rpn_results)[:, :, 1:]
        temp_ROIs = np.zeros_like(proposal_box)
        temp_ROIs[:, :, [0, 1, 2, 3]] = proposal_box[:, :, [1, 0, 3, 2]]
        classifier_pred = self.model_classifier.predict([base_layer, temp_ROIs])
        
        #-------------------------------------------------------------#
        #   The prediction frame is obtained by decoding the suggestion box by using the prediction results of classifier
        #-------------------------------------------------------------#
        results = self.bbox_util.detection_out_classifier(classifier_pred, proposal_box, self.config, self.confidence)

        if len(results[0])==0:
            return old_image
            
        results = np.array(results[0])
        boxes = results[:, :4]
        top_conf = results[:, 4]
        top_label_indices = results[:, 5]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * old_width
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * old_height

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        
        thickness = max((np.shape(old_image)[0] + np.shape(old_image)[1]) // old_width * 2, 1)

        image = old_image
        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]

            left, top, right, bottom = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def close_session(self):
        self.sess.close()
