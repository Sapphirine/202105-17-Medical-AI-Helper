import os
import random
from tqdm import tqdm

for i in tqdm(range(1000)):
    # =======================================================#
    g_root_path = "/home/gxt/study/EAD6895/trainingData_detection/rawMaterial/XML/"
    xmlfilepath = "Ann"  # Mark file storage path
    saveBasePath = "/home/gxt/study/EAD6895/trainingData_detection/rawMaterial/XML/ImageSet/"  # Imagesets information generation path
    trainval_percent = 0.8
    train_percent = 0.8
    # =======================================================#

    os.chdir(g_root_path)
    total_xml = os.listdir(xmlfilepath)
    num = len(total_xml)
    xml_list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(xml_list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("train  size", tr)
    ftrainval = open(saveBasePath + "trainval.txt", "w")
    ftest = open(saveBasePath + "test.txt", "w")
    ftrain = open(saveBasePath + "train.txt", "w")
    fval = open(saveBasePath + "val.txt", "w")

    for i in xml_list:
        name = total_xml[i][:-4] + "\n"
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    pass
