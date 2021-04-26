import os
import shutil
def fun(xml_path,image_path,dest_path):
    for f in os.listdir(xml_path):
        name=os.path.splitext(f)[0] #取出这个文件的名字
        for i in os.listdir(image_path):
            if os.path.splitext(i)[0]==name:
                dest=dest_path+i
                image=image_path+i
                shutil.copyfile(image,dest)

def main():
    xml_path="/Users/maxgeng/Study/FYP/CornerNet/data/ead/test/Annotations/"
    imagepath="/Users/maxgeng/Study/FYP/CornerNet/data/ead/picture/"
    destination="/Users/maxgeng/Desktop/test/"
    fun(xml_path,imagepath,destination)


main()