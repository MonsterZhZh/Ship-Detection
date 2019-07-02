# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

import numpy as np
import cv2
import math
import random


def short_side_resize(img_tensor, gtboxes_and_label, target_shortside_len):
    '''

    :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 9]
    :param target_shortside_len:
    :return:
    '''

    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    new_h, new_w = tf.cond(tf.less(h, w),
                           true_fn=lambda: (target_shortside_len, target_shortside_len * w//h),
                           false_fn=lambda: (target_shortside_len * h//w,  target_shortside_len))

    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    x1, y1, x2, y2, x3, y3, x4, y4, head_x, head_y, label = tf.unstack(gtboxes_and_label, axis=1)

    x1, x2, x3, x4, head_x = x1 * new_w//w, x2 * new_w//w, x3 * new_w//w, x4 * new_w//w, head_x * new_w//w
    y1, y2, y3, y4, head_y = y1 * new_h//h, y2 * new_h//h, y3 * new_h//h, y4 * new_h//h, head_y * new_h//h

    img_tensor = tf.squeeze(img_tensor, axis=0)  # ensure image tensor rank is 3
    return img_tensor, tf.transpose(tf.stack([x1, y1, x2, y2, x3, y3, x4, y4, head_x, head_y, label], axis=0))


def short_side_resize_for_inference_data(img_tensor, target_shortside_len, is_resize=True):
    h, w, = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    img_tensor = tf.expand_dims(img_tensor, axis=0)

    if is_resize:
        new_h, new_w = tf.cond(tf.less(h, w),
                               true_fn=lambda: (target_shortside_len, target_shortside_len * w // h),
                               false_fn=lambda: (target_shortside_len * h // w, target_shortside_len))
        img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    return img_tensor  # [1, h, w, c]


def flip_left_right(img_tensor, gtboxes_and_label):
    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    img_tensor = tf.image.flip_left_right(img_tensor)

    x1, y1, x2, y2, x3, y3, x4, y4, head_x, head_y, label = tf.unstack(gtboxes_and_label, axis=1)
    new_x1 = w - x1
    new_x2 = w - x2
    new_x3 = w - x3
    new_x4 = w - x4
    new_head_x = w - head_x
    return img_tensor, tf.transpose(tf.stack([new_x1, y1, new_x2, y2, new_x3, y3, new_x4, y4, new_head_x, head_y, label], axis=0))


def random_flip_left_right(img_tensor, gtboxes_and_label):

    img_tensor, gtboxes_and_label = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.5),
                                            lambda: flip_left_right(img_tensor, gtboxes_and_label),
                                            lambda: (img_tensor, gtboxes_and_label))

    return img_tensor,  gtboxes_and_label


# ******************************************Rotate img & bounding box by a random angle**********************************************

def rotate_point(img_tensor, gtboxes_and_label):

    def rotate_point_cv(img, point_label=None, keep_size = False):
        '''
        input:  1.img type:numpy_array(np.float32)  2.angle  type:int 3.point_label type:numpy_array(np.int32) shape(None, 11), last three represent head coordinates and label
        output: 1.rotated_img 2.rotated_point shape(None, 10) 
        '''
        angle = random.randint(0,90)

        if keep_size:
            cols = img.shape[1]
            rows = img.shape[0]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            img = cv2.warpAffine(img, M, (cols, rows))
            a = M[:, :2]  ##a.shape (2,2)
            b = M[:, 2:]  ###b.shape(2,1)
            b = np.reshape(b,newshape=(1,2))
            a = np.transpose(a)
            point = np.reshape(point_label[:,:-1],newshape=(len(point_label)*5,2))
            point = np.dot(point,a)+b
            point = np.reshape(point,newshape=(np.int(len(point)/5),10))
            label = point_label[:,-1]
            label = label[:, np.newaxis]
            point = np.concatenate((point, label), axis=1)
            return img, np.int32(point)
        else:
            cols = img.shape[1]
            rows = img.shape[0]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
     
            heightNew = int(cols * math.fabs(math.sin(math.radians(angle))) + rows* math.fabs(math.cos(math.radians(angle))))
            widthNew = int(rows * math.fabs(math.sin(math.radians(angle))) + cols* math.fabs(math.cos(math.radians(angle))))
            M[0, 2] += (widthNew - cols) / 2  
            M[1, 2] += (heightNew - rows) / 2 
     
            img = cv2.warpAffine(img, M, (widthNew, heightNew))
            a = M[:, :2]  ##a.shape (2,2)
            b = M[:, 2:]  ###b.shape(2,1)
            b = np.reshape(b, newshape=(1, 2))
            a = np.transpose(a)
            point = np.reshape(point_label[:,:-1], newshape=(len(point_label) * 5, 2))
            point = np.dot(point, a) + b
            point = np.reshape(point, newshape=(np.int(len(point) / 5), 10))
            label = point_label[:,-1]
            label = label[:, np.newaxis]
            point = np.concatenate((point, label), axis=1)
            return img, np.int32(point)

    rotated_img_tensor, rotated_gtboxes_and_label = tf.py_func(rotate_point_cv, inp=[img_tensor, gtboxes_and_label], Tout=[tf.float32, tf.int32])
    rotated_gtboxes_and_label.set_shape(gtboxes_and_label.get_shape())
    # rotated_img_tensor = tf.reshape(rotated_img_tensor, [-1,-1,3]) # We only consider RGB three channels now.
    rotated_img_tensor.set_shape(tf.TensorShape([None,None,3]))
    return rotated_img_tensor, rotated_gtboxes_and_label
