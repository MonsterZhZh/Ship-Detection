# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
sys.path.append('../') 

import tensorflow.contrib.slim as slim
import os
import time
from data.io.read_tfrecord import next_batch
from libs.configs import cfgs
from help_utils import tools
from libs.box_utils.show_box_in_tensor import *
from libs.box_utils.coordinate_convert import back_forward_convert, get_head_quadrant
from libs.box_utils.boxes_utils import get_horizen_minAreaRectangle, get_head

import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.GPU_GROUP


def test_rotate():
    with tf.Graph().as_default():
        with tf.name_scope('get_batch'):
            img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch = \
                next_batch(dataset_name=cfgs.DATASET_NAME,
                           batch_size=cfgs.BATCH_SIZE,
                           shortside_len=cfgs.SHORT_SIDE_LEN,
                           is_training=True)
            gtboxes_and_label, head = get_head(tf.squeeze(gtboxes_and_label_batch, 0))
            gtboxes_and_label = tf.py_func(back_forward_convert,
                                           inp=[gtboxes_and_label],
                                           Tout=tf.float32)
            gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 6])
            head_quadrant = tf.py_func(get_head_quadrant,
                                       inp=[head, gtboxes_and_label],
                                       Tout=tf.float32)
            head_quadrant = tf.reshape(head_quadrant, [-1, 1])

            gtboxes_and_label_minAreaRectangle = get_horizen_minAreaRectangle(gtboxes_and_label)

            gtboxes_and_label_minAreaRectangle = tf.reshape(gtboxes_and_label_minAreaRectangle, [-1, 5])

        with tf.name_scope('draw_gtboxes'):
            gtboxes_in_img = draw_box_with_color(img_batch, tf.reshape(gtboxes_and_label_minAreaRectangle, [-1, 5])[:, :-1],
                                                 text=tf.shape(gtboxes_and_label_minAreaRectangle)[0])

            gtboxes_rotate_in_img = draw_box_with_color_rotate(img_batch, tf.reshape(gtboxes_and_label, [-1, 6])[:, :-1],
                                                               text=tf.shape(gtboxes_and_label)[0],
                                                               head=head_quadrant)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            for i in range(650):
                img_gtboxes, img_gtboxes_rotate, img_name = sess.run([gtboxes_in_img, gtboxes_rotate_in_img, img_name_batch])
                img_gtboxes = np.squeeze(img_gtboxes, axis=0)
                img_gtboxes_rotate = np.squeeze(img_gtboxes_rotate, axis=0)

                print(i)
                cv2.imwrite(cfgs.INFERENCE_SAVE_PATH + '/{}_horizontal_fpn.jpg'.format(str(img_name[0])), img_gtboxes)
                cv2.imwrite(cfgs.INFERENCE_SAVE_PATH + '/{}_rotate_fpn.jpg'.format(str(img_name[0])), img_gtboxes_rotate)

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':

    test_rotate()