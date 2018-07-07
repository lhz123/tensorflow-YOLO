import tensorflow as tf

import os
import numpy as np
import cv2
import config as cfg
import pickle
import copy
import xml.etree.ElementTree as ET

class pascal_voc(object):
    def __init__(self,phase,rebuild=False):
        self.data_path=os.path.join('./data/pascal_voc/VOCdevkit/VOC2012')#数据路径
        self.cache_path=cfg.cache #pascal_train_get_label 路径
        #self.data_path = os.path.join('/data/pascal_voc/flower17/oxfordflower17')
        self.batch_size=cfg.batch_size
        self.image_size=cfg.image_size
        self.cell_size=cfg.cell_size
        self.classes=cfg.classes_name
        self.class_to_ind=dict(zip(self.classes,range(len(self.classes))))#将对应标签转换为字典形式
        self.phase=phase#phase= train or test 训练还是测试
        self.rebuild=rebuild
        self.epoch=3#
        self.cursor=0
        self.flipped=cfg.FLIPPED
        self.get_labels=None#get_labels为函数load_labels
        self.prepare()

    def get(self):#获取图片和标签
        images=np.zeros((self.batch_size,self.image_size,self.image_size,3))
        labels=np.zeros((self.batch_size,self.cell_size,self.cell_size,25))
        count=0
        while count < self.batch_size:
            imname=self.get_labels[self.cursor]['imname']
            flipped=self.get_labels[self.cursor]['flipped']
            images[count,:,:,:]=self.image_read(imname,flipped)
            labels[count,:,:,:]=self.get_labels[self.cursor]['label']
            count +=1
            self.cursor +=1
            if self.cursor >= len(self.get_labels):
                np.random.shuffle(self.get_labels)
                self.cursor = 0
                self.epoch += 1
        return images,labels

    def image_read(self,imname,flipped=False):#读取图片
        image = cv2.imread(imname)
        image = cv2.resize(image,(self.image_size,self.image_size))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image/255.0)*2.0-1.0
        if flipped:
            image = image[:,::-1,:]
        return image

    def prepare(self):#获取标签并打乱顺序
        get_labels=self.load_labels()
        if self.flipped:
            print('Appending horizontally-flipped training examples......')
            get_labels_cp=copy.deepcopy(get_labels)#深度复制，形成一个新的对象
            #print(get_labels_cp)
            for idx in range(len(get_labels_cp)):
                get_labels_cp[idx]['flipped']=True
                get_labels_cp[idx]['label']=get_labels_cp[idx]['label'][:,::-1,:]
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if get_labels_cp[idx]['label'][i,j,0]==1:
                            get_labels_cp[idx]['label'][i,j,1]=self.image_size - 1-get_labels_cp[idx]['label'][i,j,1]
            get_labels += get_labels_cp
        np.random.shuffle(get_labels)
        self.get_labels=get_labels
        return get_labels

    def load_labels(self):#从Pascal_train_get_labels 加载标签
        cache_file=os.path.join(self.cache_path,'pascal_'+self.phase+'_gt_labels.pkl')
        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ',cache_file)
            with open(cache_file,'rb')as f:
                get_labels=pickle.load(f)#得出的为好像是图片标签
                #print(get_labels)
            return get_labels
        print('Processing get_labels from: ' + self.data_path)
        # if not os.path.exists(self.cache_path):
        #     os.makedirs(self.cache_path)
        #     print('sdfbfgb')
        if self.phase=='train':
            txtname=os.path.join(self.data_path,'ImageSets','Main','trainval.txt')#读取训练部分图片名字
        else:
            txtname=os.path.join(self.data_path,'ImageSets','Main','test.txt')
        with open(txtname,'r') as f:
            self.image_index=[x.strip() for x in f.readlines()]
        get_labels=[]

        for index in self.image_index:
            label,num=self.load_pascal_annotation(index)#num为一张图片中标记的bounding box的个数
            if num == 0:
                continue
            imname= os.path.join(self.data_path,'JPEGImages',index+'.jpg')
            get_labels.append({'imname':imname,'label':label,'flipped':False})
            print('Saving get_labels to :' + cache_file)
            with open(cache_file,'wb')as f:
                pickle.dump(get_labels,f)

            return get_labels
    
    def load_pascal_annotation(self, index):#index为图片的编号
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])

        label = np.zeros((self.cell_size, self.cell_size, 25))
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)#解析XML
        objs = tree.findall('object')#obj的个数，一张图片上并不只有一个obj

        for obj in objs:
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, len(objs)











        
        
        
        
