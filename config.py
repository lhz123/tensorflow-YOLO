import tensorflow as tf

import os


log_path='./log/output'
event_path='./tensorboard'
cache='./data/pascal_voc/cache'
output_dir=os.path.join(log_path,'output')

classes_name=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

FLIPPED=True
image_size=448
batch_size=128
boxes_per_cell=2
cell_size=7
alpha=0.1
object_scale=1.0
noobject_scale=1.0
class_scale=2.0
coord_scale=5.0

GPU=''
learning_rate=0.0001
decay_step=3000
staircase=True
max_iter=100000
summary_iter=10
saver_iter=2000

threshold=0.2
iou_threshold=0.5

