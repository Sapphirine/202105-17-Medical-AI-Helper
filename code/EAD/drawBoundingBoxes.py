import os
import xml.dom.minidom
import cv2 as cv
 
ImgPath = '/home/gxt/study/EAD6895/trainingData_detection/try/img/'
AnnoPath = '/home/gxt/study/EAD6895/trainingData_detection/try/anno/'  #XML file address
save_path = '/home/gxt/study/EAD6895/trainingData_detection/try/output'
def draw_anchor(ImgPath,AnnoPath,save_path):
    imagelist = os.listdir(ImgPath)
    for image in imagelist:
 
        image_pre, ext = os.path.splitext(image)
        imgfile = ImgPath + image
        xmlfile = AnnoPath + image_pre + '.xml'
        print(xmlfile)
        # print(image)
        # Open XML document
        DOMTree = xml.dom.minidom.parse(xmlfile)
        # Get document element object
        collection = DOMTree.documentElement
        # Read picture
        img = cv.imread(imgfile)
 
        filenamelist = collection.getElementsByTagName("filename")
        filename = filenamelist[0].childNodes[0].data
        print(filename)
        # Get the information named object
        objectlist = collection.getElementsByTagName("object")
 
        for objects in objectlist:
            # Each object gets the information of the sub tag named name
            namelist = objects.getElementsByTagName('name')
            # Get the value of a specific name through this statement
            objectname = namelist[0].childNodes[0].data
 
            bndbox = objects.getElementsByTagName('bndbox')
            # print(bndbox)
            for box in bndbox:
                x1_list = box.getElementsByTagName('xmin')
                x1 = int(x1_list[0].childNodes[0].data)
                y1_list = box.getElementsByTagName('ymin')
                y1 = int(y1_list[0].childNodes[0].data)
                x2_list = box.getElementsByTagName('xmax')   #Pay attention to the coordinates and see if you need to convert them
                x2 = int(x2_list[0].childNodes[0].data)
                y2_list = box.getElementsByTagName('ymax')
                y2 = int(y2_list[0].childNodes[0].data)
                cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
                cv.putText(img, objectname, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0),
                           thickness=2)
                # cv.imshow('head', img)
                cv.imwrite(save_path+'/'+filename, img)   #save picture

draw_anchor(ImgPath,AnnoPath,save_path)