# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 18:35:18 2019

@author: 37112
"""

import os
impath = '/home/gxt/study/EAD6895/trainingData_detection/trainingDataResize/'
#anpath = '/home/gxt/study/EAD6895/trainingData_detection/XML/'
p = os.listdir(impath)
#p1=sorted(p)
#print(len(p1))
a=os.listdir(impath)
a1=sorted(a)
print(a1)
'''
fp1 = open('/Users/maxgeng/Study/FYP/Code/dataset/Latest/' + "fp1.txt", "w")
fa1 = open('/Users/maxgeng/Study/FYP/Code/dataset/Latest/' + "fa1.txt", "w")
for i in p1:
    name=i+"\n"
    fp1.write(name)

for j in a1:
    name=j+"\n"
    fa1.write(name)
'''

count=1
for i in a1:
    if count>=1 and count<=9:
        OldImgName = impath + i
        NewImgName = impath +'000000'+ str(count) + '.jpg'
        #OldAnnName = anpath + i
        #NewAnnName = anpath +'000000'+ str(count) + '.xml'
        os.rename(OldImgName,NewImgName)
        #os.rename(OldAnnName,NewAnnName)

    elif count>=10 and count<=99:
        OldImgName = impath + i
        NewImgName = impath +'00000'+ str(count) + '.jpg'
        #OldAnnName = anpath + i
        #NewAnnName = anpath +'00000'+ str(count) + '.xml'
        os.rename(OldImgName,NewImgName)
        #os.rename(OldAnnName,NewAnnName)
    elif count>=100 and count<=999:
        OldImgName = impath + i
        NewImgName = impath +'0000'+ str(count) + '.jpg'
        #OldAnnName = anpath + i
        #NewAnnName = anpath +'0000'+ str(count) + '.xml'
        os.rename(OldImgName,NewImgName)
        #os.rename(OldAnnName,NewAnnName)
    elif count>=1000 and count<=9999:
        OldImgName = impath + i
        NewImgName = impath +'000'+ str(count) + '.jpg'
        #OldAnnName = anpath + i
        #NewAnnName = anpath +'000'+ str(count) + '.xml'
        os.rename(OldImgName,NewImgName)
        #os.rename(OldAnnName,NewAnnName)                           
    count += 1
    print(count)

'''
count1=1
for j in a1:
    if count1>=1 and count1<=9:
        OldAnnName = anpath + j
        NewAnnName = anpath +'00000'+ str(count1) + '.xml'
        os.rename(OldAnnName,NewAnnName)

    elif count1>=10 and count1<=99:
        OldAnnName = anpath + j
        NewAnnName = anpath +'0000'+ str(count1) + '.xml'
        os.rename(OldAnnName,NewAnnName)
    elif count1>=100 and count1<=999:
        OldAnnName = anpath + j
        NewAnnName = anpath +'000'+ str(count1) + '.xml'
        os.rename(OldAnnName,NewAnnName)
    elif count1>=1000 and count1<=9999:
        OldAnnName = anpath + j
        NewAnnName = anpath +'00'+ str(count1) + '.xml'
        os.rename(OldAnnName,NewAnnName)                           
    count1 += 1
    print(NewAnnName)
'''


# =============================================================================
# 
# for i in range(0,1305):
#     oldname1 = impath + p[n]
#     oldname2 = anpath + a[n]
# #    newname1 = impath +'000000'+ str(n+1) + '.jpg'
# #    newname2 = anpath +'000000'+ str(n+1) + '.txt'
#     if n < 113:
#         newname1 = impath +'000'+ str(n+887) + '.jpg'
#         newname2 = anpath +'000'+ str(n+887) + '.txt'
#     else:
#         newname1 = impath +'00'+ str(n+887) + '.jpg'
#         newname2 = anpath +'00'+ str(n+887) + '.txt'
#     os.rename(oldname1,newname1)
#     os.rename(oldname2,newname2)
#     n += 1
# =============================================================================
