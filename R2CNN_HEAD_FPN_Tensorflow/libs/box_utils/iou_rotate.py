# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import time
from libs.box_utils.rbbox_overlaps import rbbx_overlaps
from libs.box_utils.iou_cpu import get_iou_matrix
import tensorflow as tf
from libs.box_utils.coordinate_convert import *


def iou_rotate_calculate(boxes1, boxes2, use_gpu=True, gpu_id=0):
    '''

    :param boxes_list1:[N, 8] tensor
    :param boxes_list2: [M, 8] tensor
    :return:
    '''

    boxes1 = tf.cast(boxes1, tf.float32)
    boxes2 = tf.cast(boxes2, tf.float32)
    if use_gpu:

        iou_matrix = tf.py_func(rbbx_overlaps,
                                inp=[boxes1, boxes2, gpu_id],
                                Tout=tf.float32)
    else:
        iou_matrix = tf.py_func(get_iou_matrix, inp=[boxes1, boxes2],
                                Tout=tf.float32)

    iou_matrix = tf.reshape(iou_matrix, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])

    return iou_matrix


def iou_rotate_calculate1(boxes1, boxes2, use_gpu=True, gpu_id=0):

    # start = time.time()
    if use_gpu:
        ious = rbbx_overlaps(boxes1, boxes2, gpu_id)
    else:
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        ious = []
        for i, box1 in enumerate(boxes1):
            temp_ious = []
            r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
            for j, box2 in enumerate(boxes2):
                r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

                int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)

                    int_area = cv2.contourArea(order_pts)

                    inter = int_area * 1.0 / (area1[i] + area2[j] - int_area)
                    temp_ious.append(inter)
                else:
                    temp_ious.append(0.0)
            ious.append(temp_ious)

    # print('{}s'.format(time.time() - start))

    return np.array(ious, dtype=np.float32)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '13'
    boxes1 = np.array([[50, 50, 100, 300, 0],
                       [60, 60, 100, 200, 0]], np.float32)

    boxes2 = np.array([[50, 50, 100, 300, -45.],
                       [200, 200, 100, 200, 0.]], np.float32)

    start = time.time()
    with tf.Session() as sess:
        ious = iou_rotate_calculate1(boxes1, boxes2, use_gpu=False)
        print(sess.run(ious))
        print('{}s'.format(time.time() - start))

    # start = time.time()
    # for _ in range(10):
    #     ious = rbbox_overlaps.rbbx_overlaps(boxes1, boxes2)
    # print('{}s'.format(time.time() - start))
    # print(ious)

    # print(ovr)



