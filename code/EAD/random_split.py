# coding=utf-8
import os, random, shutil
def moveFile(fileDir):
    pathDir = os.listdir(fileDir)  # Get the original path of the picture
    filenumber = len(pathDir)
    rate = 0.5 # Customize the proportion of pictures extracted, for example, 100 pieces and 10 pieces, which is 0.1
    picknumber = int(filenumber * rate)  # Take a certain number of pictures from the folder according to the rate ratio
    sample = random.sample(pathDir, picknumber)  # Randomly select the number of sample pictures of picknumber
    for name in sample:
        shutil.move(os.path.join(fileDir,name),os.path.join(tarDir,name))
        print(name)
    return

if __name__ == '__main__':
    tarDir = '/home/gxt/study/EAD6895/trainingData_detection/ValXML'   # Move to new folder path
    ori_path = '/home/gxt/study/EAD6895/trainingData_detection/XML' # Folder path of the initial train
    #for firstPath in os.listdir(ori_path):
    #    fileDir = os.path.join(ori_path, firstPath)  # Original picture folder path
    moveFile(ori_path)
