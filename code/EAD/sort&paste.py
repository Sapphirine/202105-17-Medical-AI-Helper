import os
import shutil
def fun(xml_path,image_path,dest_path):
    for f in os.listdir(xml_path):
        name=os.path.splitext(f)[0] #取出这个文件的名字
        for i in os.listdir(image_path):
            if os.path.splitext(i)[0]==f:
                shutil.copyfile(i,dest_path)

def main():
    xml_path=input('Enter the XML directory')
    print('The directory I chose is:  %s' % xml_path)
    imagepath=input('Enter the path of the picture library')
    destination=input('Enter the path to export the picture')
    fun(xml_path,imagepath,destination)

if __name__=='_main_':
    main()