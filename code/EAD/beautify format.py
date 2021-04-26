# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 20:22:12 2018

@author: 37112
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 14:45:44 2018

@author: 37112
"""

def prettyXml(element, indent, newline, level = 0): #Elemnt is the incoming element class. The parameter indent is used for indentation and newline is used for line feed  
    if element:  #Determine whether the element has child elements
        if element.text == None or element.text.isspace(): #If the text of element has no content 
            element.text = newline + indent * (level + 1)    
        else:  
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)  
    #else:  #If the comment is removed from the two lines, the text of the element will start another line 
        #element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level  
    temp = list(element) # Convert elemnt to list
    for subelement in temp:  
        if temp.index(subelement) < (len(temp) - 1): # If it is not the last element of the list, the next line is the beginning of the same level element, and the indentation should be consistent  
            subelement.tail = newline + indent * (level + 1)  
        else:  # If it is the last element of the list, the next line is the end of the parent element, and the indentation should be one less
            subelement.tail = newline + indent * level  
        prettyXml(subelement, indent, newline, level = level + 1) # Recursive operation on child elements
          
from xml.etree import ElementTree      #Import elementtree module
import os

path = 'E:/course/6680/ead2019_trainingData-I/ead2019_trainingData-I/train_release2_task1/new/'
# 图片名
fp = os.listdir(path)
for i in range(0,1305):
    infile = path + fp[i]
    outfile = os.path.splitext(fp[i])[0] +".xml"
#    print(outfile)
    try:
        tree = ElementTree.parse(infile)   #analysis test.xml The content of this document is as above
    except FileNotFoundError:
        print("error")
        continue
    root = tree.getroot()                  #Get the root element, element class 
#    prettyXml(root, '\t', '\n')            #Implementation of beautification methods  
    prettyXml(root, '\t', '\n')            #Implementation of beautification methods 
#ElementTree.dump(root)                 #Display the beautified XML content
    tree.write('E:/course/6680/ead2019_trainingData-I/ead2019_trainingData-I/train_release2_task1/new/' + outfile)

