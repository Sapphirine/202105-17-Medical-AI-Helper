# 202105-17-Medical-AI-Helper
This is the final project of EECS E6895 in Spring 2021. This project is done by Xiaotian Geng and Chaoran Wei.
# Organization of our project
```
.
├── README.md
├── code
│   ├── Bone Suppression
│   │   ├── test_GAN
│   │   └── train_GAN
│   ├── CT
│   │   ├── networks
│   │   └── scripts
│   ├── EAD
│   │   ├── #datasetPrepariation.py
│   │   ├── ImageSet.py
│   │   ├── ImageSets.py
│   │   ├── README.md
│   │   ├── ReadTxt.py
│   │   ├── Script.code-workspace
│   │   ├── XmlModify.py
│   │   ├── XmlModifypy
│   │   ├── annotation.xml
│   │   ├── beautify format.py
│   │   ├── drawBoundingBoxes.py
│   │   ├── ga_faster_rcnn_x101_32x4d_fpn_1x_augmentated.txt
│   │   ├── modified.py
│   │   ├── random_split.py
│   │   ├── read_pkl.py
│   │   ├── rename-file.py
│   │   ├── rename.py
│   │   ├── sort&paste.py
│   │   ├── sort-list.py
│   │   ├── sort_paste.py
│   │   ├── xml_to_coco.py
│   │   ├── xml_to_coco_another_version
│   │   └── xml_to_coco_another_version.py
│   ├── README.md
│   ├── X-ray
│   │   ├── networks
│   │   └── scripts
│   └── segmentation
│       ├── brain
│       └── lung
├── tree command.ipynb
└── web app
    ├── Endosopic.py
    ├── README.md
    ├── boneSuppression.py
    ├── brainTumor.py
    ├── ctAPP.py
    ├── frcnn.py
    ├── lungSegmentation.py
    ├── main.py
    ├── model
    ├── static
    │   ├── AI.jpg
    │   ├── BoneSuppression
    │   ├── EAD
    │   ├── Image_No_Pred_MJRoBot.png
    │   ├── Image_Prediction.png
    │   ├── MJRoBot_Avatar_1.jpg
    │   ├── MRI.jpg
    │   ├── boneSuppression.png
    │   ├── brain
    │   ├── ct.png
    │   ├── ct_analisys
    │   ├── ct_img
    │   ├── endoscopy.png
    │   ├── head.jpg
    │   ├── healthcare-collage.jpg
    │   ├── lungInfection.jpg
    │   ├── lungSegmentation
    │   ├── robot_Y_N.png
    │   ├── style.css
    │   ├── x-ray.png
    │   └── xray
    ├── templates
    │   ├── Brain.html
    │   ├── CT.html
    │   ├── EAD.html
    │   ├── LungSeg.html
    │   ├── Welcome.html
    │   ├── Xray.html
    │   └── boneSuppression.html
    └── xray.py

25 directories, 56 files
```
# Descriptions of our project
Our project contains two parts. Firstly, we achieve six functions: COVID-19 X-ray classification, COVID-19 CT classification, bone suppression, COVID-19 lung segmentation, Brain Tumor segmentation and Endoscopy Artifact Detection. The code of each function is placed in a sub file under the code folder. Secondly, we deploy all the functions to the web-app so that users could utilize our models to do medical analysis. The code is in the web app folder.