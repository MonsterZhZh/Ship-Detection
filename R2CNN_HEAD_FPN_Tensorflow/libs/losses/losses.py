# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops


def l1_smooth_losses(predict_boxes, gtboxes, object_weights, classes_weights=None):
    '''

    :param predict_boxes: [minibatch_size, -1]
    :param gtboxes: [minibatch_size, -1]
    :param object_weights: [minibatch_size, ]. 1.0 represent object, 0.0 represent others(ignored or background)
    :return:
    '''

    diff = predict_boxes - gtboxes
    abs_diff = tf.cast(tf.abs(diff), tf.float32)

    if classes_weights is None:
        '''
        first_stage:
        predict_boxes :[minibatch_size, 4]
        gtboxes: [minibatchs_size, 4]
        '''
        anchorwise_smooth_l1norm = tf.reduce_sum(
            tf.where(tf.less(abs_diff, 1), 0.5 * tf.square(abs_diff), abs_diff - 0.5),
            axis=1) * object_weights
    else:
        '''
        fast_rcnn:
        predict_boxes: [minibatch_size, 4*num_classes]
        gtboxes: [minibatch_size, 4*num_classes]
        classes_weights : [minibatch_size, 4*num_classes]
        '''
        anchorwise_smooth_l1norm = tf.reduce_sum(
            tf.where(tf.less(abs_diff, 1), 0.5*tf.square(abs_diff)*classes_weights,
                     (abs_diff - 0.5)*classes_weights),
            axis=1)*object_weights
    return tf.reduce_mean(anchorwise_smooth_l1norm, axis=0)  # reduce mean


def weighted_softmax_cross_entropy_loss(predictions, labels, weights):
    '''

    :param predictions:
    :param labels:
    :param weights: [N, ] 1 -> should be sampled , 0-> not should be sampled
    :return:
    # '''
    per_row_cross_ent = tf.nn.softmax_cross_entropy_with_logits(logits=predictions,
                                                                labels=labels)

    weighted_cross_ent = tf.reduce_sum(per_row_cross_ent * weights)
    return weighted_cross_ent / tf.reduce_sum(weights)

def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    """Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)

def test_smoothl1():

    predict_boxes = tf.constant([[1, 1, 2, 2],
                                [2, 2, 2, 2],
                                [3, 3, 3, 3]])
    gtboxes = tf.constant([[1, 1, 1, 1],
                          [2, 1, 1, 1],
                          [3, 3, 2, 1]])

    loss = l1_smooth_losses(predict_boxes, gtboxes, [1, 1, 1])

    with tf.Session() as sess:
        print(sess.run(loss))

if __name__ == '__main__':
    test_smoothl1()
