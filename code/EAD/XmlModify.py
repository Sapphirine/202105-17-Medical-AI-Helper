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

'''
#Label path
annpath = '/Users/maxgeng/Study/FYP/Code/dataset/Latest/LatestAnnotations/'
# Tag name
a = os.listdir(annpath)
#n = 0  #Picture and label subscript
a1=sorted(a)
#print(a1)
for i in a1:
    name = os.path.splitext(i)[0]
    img_name = name+'.jpg'  

    tree= ET.parse(annpath+i)
    root = tree.getroot()
    root[1].text = img_name
import xml.etree.ElementTree as ET

'''
def change_xml(xml_path):
    filelist = os.listdir(xml_path)
    filelist=sorted(filelist)
    print(filelist)
    # Open XML document
    for xmlfile in filelist:
        doc = ET.parse(xml_path+xmlfile)
        root = doc.getroot()
        sub1 = root.find('filename')  #Find the file name tag,
        name = os.path.splitext(xmlfile)[0]+'.jpg'
        sub1.text = name     #Modify label content
        
        doc.write(xml_path+xmlfile)   #Save changes

change_xml('/Users/maxgeng/Study/FYP/Code/dataset/endoscopy-artefact-detection-_ead_-dataset/xml_listedCopy/') 