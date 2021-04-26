# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 21:16:24 2019

@author: 37112
"""
import os
import cv2
import numpy as np
import xml.etree.ElementTree
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


#image path
picpath = '/Volumes/study/study/FYP/dataset/endoscopy-artefact-detection-_ead_-dataset/picture/'
#label path
annpath = '/Volumes/study/study/FYP/dataset/endoscopy-artefact-detection-_ead_-dataset/labels/'
# image name
fp = os.listdir(picpath)
# label name 
fa = os.listdir(annpath)
#n = 0  #Picture and label subscript


for i in range(0,4017):
    img_name = picpath + fp[i] # get images
#    print("img_name is",img_name)
    annotation_name = annpath + fa[i]  #get labels
#    print("annotation_name is",annotation_name)
    # Get image length and width
    img = cv2.imread(img_name,1)
#    print("img is",img)
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
#    print("height:",height)
#    print("weight:",weight)

    txtData = open(annotation_name,'rb')   # TXT text
    lineNum = len(txtData.read().splitlines()) # Number of rows in TXT
    # Save TXT content into numpy
    anndata = np.loadtxt(annotation_name)
    # Start modifying XML content
    tree = ET.parse("annotation.xml")
    root = tree.getroot()
    root[1].text = fp[i]    #Modify the content of the file name in the XML file
    root[4].find('width').text = str(width)   # Modify width
    root[4].find('height').text = str(height)  # Modify height
    # Modify the content in the object
    if lineNum == 1:
        xmin = int(anndata[1]*width)-int(anndata[3]*width/2)
        ymin = int(anndata[2]*height)-int(anndata[4]*height/2)
        xmax = int(anndata[1]*width)+int(anndata[3]*width/2)
        ymax = int(anndata[2]*height)+int(anndata[4]*height/2)
        if xmin <= 0:
            xmin = 1
        if ymin <= 0:
            ymin = 1
        if xmax > width:
            xmax = width
        if ymax > height:
            ymax = height
        for bndbox in root[6].findall('bndbox'):
            bndbox.find('xmin').text = str(xmin)
            bndbox.find('ymin').text = str(ymin)
            bndbox.find('xmax').text = str(xmax)
            bndbox.find('ymax').text = str(ymax)
        # Judgment type
        if int(anndata[0]) == 0:
            root[6].find('name').text = "specularity"
        elif int(anndata[0]) == 1:
            root[6].find('name').text = "saturation"
        elif int(anndata[0]) == 2:
            root[6].find('name').text = "artifact"
        elif int(anndata[0]) == 3:
            root[6].find('name').text = "blur"
        elif int(anndata[0]) == 4:
            root[6].find('name').text = "contrast"
        elif int(anndata[0]) == 5:
            root[6].find('name').text = "bubbles"
        elif int(anndata[0]) == 6:
            root[6].find('name').text = "instrument"
        elif int(anndata[0]) == 7:
            root[6].find('name').text = "blood"
    else:
        for a in range(lineNum):
            # Calculating boundary points of bounding box
            xmin = int(anndata[a][1]*width)-int(anndata[a][3]*width/2)
            ymin = int(anndata[a][2]*height)-int(anndata[a][4]*height/2)
            xmax = int(anndata[a][1]*width)+int(anndata[a][3]*width/2)
            ymax = int(anndata[a][2]*height)+int(anndata[a][4]*height/2)
            if xmin <= 0:
                xmin = 1
            if ymin <= 0:
                ymin = 1
            if xmax > width:
                xmax = width
            if ymax > height:
                ymax = height
            if a == 0:
                for bndbox in root[6].findall('bndbox'):
                    bndbox.find('xmin').text = str(xmin)
                    bndbox.find('ymin').text = str(ymin)
                    bndbox.find('xmax').text = str(xmax)
                    bndbox.find('ymax').text = str(ymax)
                # Judgment type
                if int(anndata[a][0]) == 0:
                    root[6].find('name').text = "specularity"
                elif int(anndata[a][0]) == 1:
                    root[6].find('name').text = "saturation"
                elif int(anndata[a][0]) == 2:
                    root[6].find('name').text = "artifact"
                elif int(anndata[a][0]) == 3:
                    root[6].find('name').text = "blur"
                elif int(anndata[a][0]) == 4:
                    root[6].find('name').text = "contrast"
                elif int(anndata[a][0]) == 5:
                    root[6].find('name').text = "bubbles"
                elif int(anndata[a][0]) == 6:
                    root[6].find('name').text = "instrument"
                else:
                    print("there are some mistake in type!!")
            else:
                object = ET.Element('object')
                root.append(object)
                name = ET.Element('name')
                if int(anndata[a][0]) == 0:
                    name.text = "specularity"
                elif int(anndata[a][0]) == 1:
                    name.text = "saturation"
                elif int(anndata[a][0]) == 2:
                    name.text = "artifact"
                elif int(anndata[a][0]) == 3:
                    name.text = "blur"
                elif int(anndata[a][0]) == 4:
                    name.text = "contrast"
                elif int(anndata[a][0]) == 5:
                    name.text = "bubbles"
                elif int(anndata[a][0]) == 6:
                    name.text = "instrument"
                else:
                    print("there are some mistake in type!!")
                object.append(name)
                pose = ET.Element('pose')
                pose.text = "Unspecified"
                object.append(pose)
                truncated = ET.Element('truncated')
                truncated.text = "0"
                object.append(truncated)
                difficult = ET.Element('difficult')
                difficult.text = "0"
                object.append(difficult)
                bndbox = ET.Element('bndbox')
                object.append(bndbox)
                xmin1 = ET.Element('xmin')
                xmin1.text = str(xmin)
                bndbox.append(xmin1)
                ymin1 = ET.Element('ymin')
                ymin1.text = str(ymin)
                bndbox.append(ymin1)
                xmax1 = ET.Element('xmax')
                xmax1.text = str(xmax)
                bndbox.append(xmax1)
                ymax1 = ET.Element('ymax')
                ymax1.text = str(ymax)
                bndbox.append(ymax1)

    outfile = os.path.splitext(fa[i])[0] +".xml"
    tree.write('E:/course/6680/ead2019_trainingData-I/ead2019_trainingData-I/train_release2_task1/new/' + outfile)
#    n += 1

#print(anndata)
