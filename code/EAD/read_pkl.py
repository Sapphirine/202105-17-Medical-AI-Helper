import pickle as p
f = open('/Users/maxgeng/mmdetection/result/ga_faster_rcnn_x101_32x4d_fpn_1x_augmentated.pkl')
inf = p.load
inf=str(inf)
ft = open('ga_faster_rcnn_x101_32x4d_fpn_1x_augmentated.txt', 'w')  
ft.write(inf)  