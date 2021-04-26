
import os
import shutil
path='/Volumes/study/study/FYP/dataset/endoscopy-artefact-detection-_ead_-dataset/trainingData_detection/'
    

#Get all the files in the directory and save them in the list
fileList=os.listdir(path)

for i in fileList:
    #if i[-1]=='g':
    #    from_path = os.path.join('/Volumes/study/study/FYP/dataset/endoscopy-artefact-detection-_ead_-dataset/trainingData_detection/', i)
    #    to_path='/Volumes/study/study/FYP/dataset/endoscopy-artefact-detection-_ead_-dataset/picture/'
        
    if i[-1]=='t':
        from_path = os.path.join('/Volumes/study/study/FYP/dataset/endoscopy-artefact-detection-_ead_-dataset/trainingData_detection/', i)
        to_path='/Volumes/study/study/FYP/dataset/endoscopy-artefact-detection-_ead_-dataset/labels/'

        shutil.copy(from_path,to_path)
            
    print(i)