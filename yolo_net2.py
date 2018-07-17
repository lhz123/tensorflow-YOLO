import tensorflow as tf
import os
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d, global_avg_pool
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge

slim = tf.contrib.slim
import config as cfg


# 同一个类内的函数之间的调用部分前后顺序

class YOLONet(object):
    def __init__(self):
        self.classes = cfg.classes_name
        self.num_classes = len(self.classes)
        self.image_size = cfg.image_size
        self.cell_size = cfg.cell_size
        self.boxes_per_cell = cfg.boxes_per_cell
        self.output_size = (self.cell_size * self.cell_size) * (
        self.num_classes + self.boxes_per_cell * 5)  # 7*7*(20+2*5)
        self.boundary1 = self.cell_size * self.cell_size * self.num_classes
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell
        self.object_scale = cfg.object_scale
        self.noobject_scale = cfg.noobject_scale
        self.class_scale = cfg.class_scale
        self.coord_scale = cfg.coord_scale
        self.learning_rate = cfg.learning_rate
        self.batch_size = cfg.batch_size

        self.offset = np.transpose(
            np.reshape(np.array([np.arange(self.cell_size * self.cell_size * self.boxes_per_cell)]),
                       (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        self.logits = self.build_network(self.images, num_output=self.output_size,width_multiplier=1)

        # if is_training:
        self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, 5 + self.num_classes])
        self.loss_layer(self.logits, self.labels)
        self.total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('total loss', self.total_loss)

    def _depthwise_separable_conv(self,inputs,
                                  num_pwc_filters,
                                  width_multiplier,
                                  sc,
                                  downsample=False):

        num_pwc_filters = round(num_pwc_filters * width_multiplier)#返回浮点书的四舍五入值
        _stride = 2 if downsample else 1

        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      stride=_stride,
                                                      depth_multiplier=1,
                                                      kernel_size=[3, 3],
                                                      scope=sc + '/depthwise_conv')

        bn = slim.batch_norm(depthwise_conv, scope=sc + '/dw_batch_norm')
        pointwise_conv = slim.convolution2d(bn,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            scope=sc + '/pointwise_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_batch_norm')
        return bn
    def build_network(self, images, num_output, width_multiplier,scope='yolo'):
        with tf.variable_scope(scope) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                                activation_fn=None,
                                outputs_collections=[end_points_collection]):
                with slim.arg_scope([slim.batch_norm],
                                    activation_fn=tf.nn.relu,
                                    fused=True):
                    net = slim.convolution2d(images, round(32 * width_multiplier), [3, 3], stride=2, padding='SAME',
                                             scope='conv_1')
                    net = slim.batch_norm(net, scope='conv_1/batch_norm')
                    net = self._depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
                    net = self._depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
                    net = self._depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
                    net = self._depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5')
                    net = self._depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
                    net = self._depthwise_separable_conv(net, 512, width_multiplier, downsample=True, sc='conv_ds_7')

                    net = self._depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8')
                    net = self._depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
                    net = self._depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
                    net = self._depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
                    net = self._depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')

                    net = self._depthwise_separable_conv(net, 1024, width_multiplier, downsample=True, sc='conv_ds_13')
                    net = self._depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14')
                    net = global_avg_pool(net)

            # end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            # net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            # end_points['squeeze'] = net
            logits = slim.fully_connected(net, num_output, activation_fn=None, scope='fc_16')
            #predictions = slim.softmax(logits, scope='Predictions')

        return logits

    # 计算IOU,即预测bounding box 与标签的bounding box 的重叠部分
    def calc_iou(self, boxes1, boxes2, scope='iou'):
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

    # 损失函数定义
    def loss_layer(self, predict, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(predict[:, :self.boundary1], [self.batch_size, self.cell_size, self.cell_size,
                                                                       self.num_classes])
            predict_scales = tf.reshape(predict[:, self.boundary1:self.boundary2], [self.batch_size, self.cell_size,
                                                                                    self.cell_size,
                                                                                    self.boxes_per_cell])
            predict_boxes = tf.reshape(predict[:, self.boundary2:], [self.batch_size, self.cell_size,
                                                                     self.cell_size, self.boxes_per_cell, 4])

            response = tf.reshape(labels[..., 0], [self.batch_size, self.cell_size, self.cell_size, 1])
            boxes = tf.reshape(labels[..., 1:5], [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[..., 5:]

            # 偏置的shape（7,7,2）
            offset = tf.constant(self.offset, dtype=tf.float32)
            # shape为（1,7,7,2）
            offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            # shape为（45,7，7,2）
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])  # 对矩阵进行复制
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





