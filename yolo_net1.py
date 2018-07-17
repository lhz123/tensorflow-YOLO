import tensorflow as tf
import os
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d,global_avg_pool
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
slim = tf.contrib.slim
import config as cfg

#同一个类内的函数之间的调用部分前后顺序

class YOLONet(object):
    def __init__(self):
        self.classes=cfg.classes_name
        self.num_classes=len(self.classes)
        self.image_size=cfg.image_size
        self.cell_size=cfg.cell_size
        self.boxes_per_cell=cfg.boxes_per_cell
        self.output_size=(self.cell_size*self.cell_size)*(self.num_classes+self.boxes_per_cell*5)#7*7*(20+2*5)
        self.boundary1=self.cell_size*self.cell_size*self.num_classes
        self.boundary2=self.boundary1+self.cell_size*self.cell_size*self.boxes_per_cell
        self.object_scale=cfg.object_scale
        self.noobject_scale=cfg.noobject_scale
        self.class_scale=cfg.class_scale
        self.coord_scale=cfg.coord_scale
        self.learning_rate=cfg.learning_rate
        self.batch_size=cfg.batch_size

        self.offset=np.transpose(np.reshape(np.array([np.arange(self.cell_size*self.cell_size*self.boxes_per_cell)]),
                                 (self.boxes_per_cell,self.cell_size,self.cell_size)),(1,2,0))
        self.images=tf.placeholder(tf.float32,[None,self.image_size,self.image_size,3],name='images')
        self.logits=self.build_network(self.images,num_output=self.output_size)

        # if is_training:
        self.labels=tf.placeholder(tf.float32,[None,self.cell_size,self.cell_size,5+self.num_classes])
        self.loss_layer(self.logits,self.labels)
        self.total_loss=tf.losses.get_total_loss()
        tf.summary.scalar('total loss',self.total_loss)

    def entry_flow(self,input_data):
        net = slim.conv2d(input_data, 32, [3, 3], 2)
        net1 = slim.conv2d(net, 64, [3, 3])
        net1_1 = slim.conv2d(net1, 128, [1, 1], 2)

        net = slim.separable_conv2d(net1, 128, [3, 3], depth_multiplier=1)
        net = slim.separable_conv2d(net, 128, [3, 3], depth_multiplier=1)

        net = slim.max_pool2d(net, [2, 2], 2)

        net2 = tf.add(net, net1_1)  # 对应元素相加

        net2_2 = slim.conv2d(net2, 256, [2, 2], 2)

        net = slim.separable_conv2d(net2, 256, [3, 3], depth_multiplier=1)
        net = slim.separable_conv2d(net, 256, [3, 3], depth_multiplier=1)

        net = slim.conv2d(net, 256, [2, 2], 2)

        net3 = tf.add(net, net2_2)
        net3_3 = slim.conv2d(net3, 256, [1, 1], 2)

        net = slim.separable_conv2d(net3, 256, [3, 3], depth_multiplier=1)
        net = slim.separable_conv2d(net, 256, [3, 3], depth_multiplier=1)
        net = slim.conv2d(net, 256, [1, 1], 2)
        net3 = tf.add(net, net3_3)

        return net3

    def middle_flow(self,input_data):
        net = slim.separable_conv2d(input_data, 728, [3, 3], depth_multiplier=1)
        net = slim.separable_conv2d(net, 728, [3, 3], depth_multiplier=1)
        net = slim.separable_conv2d(net, 256, [3, 3], depth_multiplier=1)
        net = tf.add(net, input_data)
        return net

    def exit_flow(self,input_data, num_classes):
        net = slim.separable_conv2d(input_data, 728, [3, 3], depth_multiplier=1)
        net = slim.separable_conv2d(net, 1024, [3, 3], depth_multiplier=1)

        net = slim.conv2d(net, 1024, [2, 2], 2)

        net1 = slim.conv2d(input_data, 1024, [1, 1], 2)
        net = tf.add(net, net1)
        net = slim.separable_conv2d(net, 1536, [3, 3], depth_multiplier=1)
        net = slim.separable_conv2d(net, 2048, [3, 3], depth_multiplier=1)
        net = global_avg_pool(net)
        net = slim.fully_connected(net, num_classes)

        net = slim.softmax(net)
        return net

    def build_network(self,images,num_output,scope='yolo'):
        #net = conv_2d(images, 3, 1, strides=1, activation='relu', name='layer1')
        net = self.entry_flow(images)
        for i in range(6):
            net = self.middle_flow(net)

        net = self.exit_flow(net, num_output)
        return net


    #计算IOU,即预测bounding box 与标签的bounding box 的重叠部分
    def calc_iou(self,boxes1,boxes2,scope='iou'):

        with tf.variable_scope(scope):
            boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                               boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
            boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

            boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                               boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
            boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
            rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

            # calculate the boxs1 square and boxs2 square
            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                      (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                      (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


    #损失函数定义
    def loss_layer(self,predict,labels,scope='loss_layer'):
        with tf.variable_scope(scope):
            predict_classes=tf.reshape(predict[:,:self.boundary1],[self.batch_size, self.cell_size, self.cell_size,
                                                                   self.num_classes])
            predict_scales=tf.reshape(predict[:,self.boundary1:self.boundary2],[self.batch_size,self.cell_size,
                                                                               self.cell_size,self.boxes_per_cell])
            predict_boxes=tf.reshape(predict[:,self.boundary2:],[self.batch_size,self.cell_size,
                                                                               self.cell_size,self.boxes_per_cell,4])

            response=tf.reshape(labels[...,0],[self.batch_size,self.cell_size,self.cell_size,1])
            boxes=tf.reshape(labels[...,1:5],[self.batch_size,self.cell_size,self.cell_size,1,4])
            boxes=tf.tile(boxes,[1,1,1,self.boxes_per_cell,1])/self.image_size
            classes=labels[...,5:]

            # 偏置的shape（7,7,2）
            offset = tf.constant(self.offset, dtype=tf.float32)
            # shape为（1,7,7,2）
            offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            # shape为（45,7，7,2）
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])#对矩阵进行复制
            # shape为（4,45,7,7,2）
            predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                                           (predict_boxes[:, :, :, :, 1] + tf.transpose(offset,
                                                                                        (0, 2, 1, 3))) / self.cell_size,
                                           tf.square(predict_boxes[:, :, :, :, 2]),
                                           tf.square(predict_boxes[:, :, :, :, 3])])
            # shape为（45,7,7,2,4）
            predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])

            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack([boxes[:, :, :, :, 0] * self.cell_size - offset,
                                   boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                                   tf.sqrt(boxes[:, :, :, :, 2]),
                                   tf.sqrt(boxes[:, :, :, :, 3])])
            boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                                        name='class_loss') * self.class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                                         name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                                           name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                                        name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
            tf.summary.histogram('iou', iou_predict_truth)





