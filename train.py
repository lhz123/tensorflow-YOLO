import tensorflow as tf
import numpy as np
import config as cfg
from read_data import pascal_voc
from yolo_net import YOLONet
import datetime
import os
import argparse
from timer import Timer
slim=tf.contrib.slim

class Solver(object):

    def __init__(self,net,data):
        self.net=net
        self.data=data
        #self.weight_file=cfg.log_path
        self.event_file=cfg.event_path#tensorboard path
        self.max_iter=cfg.max_iter #最大迭代步数
        self.ini_learning_rate=cfg.learning_rate#学习率
        self.decay_step=cfg.decay_step#权重衰减率
        self.staircase_iter=cfg.staircase# straircase=True
        self.summary_iter=cfg.summary_iter #tensorboard 保存的步数
        self.saver_iter=cfg.saver_iter   #参数保存步数
        self.output_dir=os.path.join(cfg.output_dir)#参数保存路径
        if not os.path.exists(self.output_dir):#若路径不存在则生成路径
            os.makedirs(self.output_dir)
        #self.save_cfg()
        self.variable_to_restore=tf.global_variables()
        self.saver=tf.train.Saver(self.variable_to_restore)
        self.summary_op=tf.summary.merge_all()
        self.writer=tf.summary.FileWriter(self.event_file,flush_secs=60)

        self.global_step=tf.train.create_global_step()
        self.learning_rate=tf.train.exponential_decay(self.ini_learning_rate,self.global_step,self.decay_step,
                                                      self.staircase_iter,name='learning_rate')
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate)#优化函数
        self.train_op=slim.learning.create_train_op(self.net.total_loss,self.optimizer,global_step=self.global_step)
        #进行训练优化

        # gpu_options=tf.GPUOptions()
        # config=tf.ConfigProto(gpu_option=gpu_options)
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.writer.add_graph(self.sess.graph)
    #训练函数
    def train(self):

        train_timer=Timer()

        load_timer=Timer()

        for step in range(1,self.max_iter+1):
            load_timer.tic()

            images,labels=self.data.get()#self.data就是main函数中的pascal，而get就是read_data中的get函数
            load_timer.toc()

            feed_dict={self.net.images:images,self.net.labels:labels}

            # if step % self.summary_iter == 0:
            if step % (self.summary_iter ) ==0:

                train_timer.tic()

                summary_str,loss,_=self.sess.run([self.summary_op,self.net.total_loss,self.train_op],
                                                 feed_dict=feed_dict)

                train_timer.toc()

                print('step=',int(step))
                print('loss=',loss)
                self.writer.add_summary(summary_str,step)#每隔1000步保存一次tensorboard文件
            if step % self.saver_iter ==0:#每隔1000步保存一次参数文件
                self.saver.save(self.sess,self.output_dir,global_step=self.global_step)


def main():

    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',default='',type=str)
    args=parser.parse_args()

    if args.gpu is not None:
        cfg.GPU=args.gpu

    os.environ['CUDA_VISIBLE_DEVICES']=cfg.GPU

    yolo=YOLONet()#调用yolo_net中的YOLONet类
    pascal=pascal_voc('train')#调用read_data中的pascal_voc类
    solver=Solver(yolo,pascal)#yolo=net,pascal=data

    print('star training......')
    solver.train()#进行训练
    print('training finished')

if __name__=='__main__':
    main()


