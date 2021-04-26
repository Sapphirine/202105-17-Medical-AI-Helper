import os
import shutil
for line in open("/Users/maxgeng/Study/FYP/Code/dataset/theFianlVocDateset/ImageSet/train.txt"): 
    target=line[0:-1]
    for i in os.listdir("/Users/maxgeng/Study/FYP/Code/dataset/theFianlVocDateset/Ann/"):
            name=os.path.splitext(i)[0]
            if name==target:
                print(name)
                path_anno="/Users/maxgeng/Study/FYP/Code/dataset/theFianlVocDateset/Ann/"+i
                dest_anno="/Users/maxgeng/Study/FYP/Code/dataset/theFianlVocDateset/TrainAnn/"+i
                #path_img="/Users/maxgeng/Study/FYP/Code/dataset/theFianlVocDateset/Img/"+name+".jpg"
                #dest_img="/Users/maxgeng/Study/FYP/Code/dataset/theFianlVocDateset/ValImg/"+name+".jpg"
                shutil.copyfile(path_anno,dest_anno)
                #shutil.copyfile(path_img,dest_img)

