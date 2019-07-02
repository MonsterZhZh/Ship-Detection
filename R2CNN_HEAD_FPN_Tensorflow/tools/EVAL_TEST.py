# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
sys.path.append('../')

import tensorflow as tf
import os
import time
from data.io.read_tfrecord import next_batch
from libs.networks.network_factory import get_network_byname
from libs.label_name_dict.label_dict import *
from libs.rpn import build_rpn
from help_utils.tools import *
from tools import restore_model
import pickle
from libs.box_utils.coordinate_convert import *
from libs.box_utils.boxes_utils import get_horizen_minAreaRectangle, get_head
from libs.fast_rcnn import build_fast_rcnn
from libs.box_utils import iou_rotate

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def make_dict_packle(_gtboxes_and_label, _gtboxes_and_label_minAreaRectangle):

    gtbox_r_list = []
    gtbox_h_list = []

    for i, box_h in enumerate(_gtboxes_and_label_minAreaRectangle):
        bbox_dict = {}
        ymin, xmin, ymax, xmax = box_h[0], box_h[1], box_h[2], box_h[3]
        bbox_dict['bbox'] = np.array([xmin, ymin, xmax, ymax], np.float64)
        bbox_dict['name'] = LABEl_NAME_MAP[int(_gtboxes_and_label_minAreaRectangle[i, -1])]
        gtbox_h_list.append(bbox_dict)

    for j, box_r in enumerate(_gtboxes_and_label):
        bbox_dict = {}
        x_c, y_c, w, h, theta = box_r[1], box_r[0], box_r[3], box_r[2], box_r[4]
        bbox_dict['bbox'] = np.array([x_c, y_c, w, h, theta], np.float64)
        bbox_dict['name'] = LABEl_NAME_MAP[int(_gtboxes_and_label[j, -1])]
        gtbox_r_list.append(bbox_dict)

    return gtbox_h_list, gtbox_r_list


def eval_ship(img_num):
    with tf.Graph().as_default():

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

        # ***********************************************************************************************
        # *                                         share net                                           *
        # ***********************************************************************************************
        _, share_net = get_network_byname(net_name=cfgs.NET_NAME,
                                          inputs=img_batch,
                                          num_classes=None,
                                          is_training=True,
                                          output_stride=None,
                                          global_pool=False,
                                          spatial_squeeze=False)

        # ***********************************************************************************************
        # *                                            RPN                                              *
        # ***********************************************************************************************
        rpn = build_rpn.RPN(net_name=cfgs.NET_NAME,
                            inputs=img_batch,
                            gtboxes_and_label=None,
                            is_training=False,
                            share_head=cfgs.SHARE_HEAD,
                            share_net=share_net,
                            stride=cfgs.STRIDE,
                            anchor_ratios=cfgs.ANCHOR_RATIOS,
                            anchor_scales=cfgs.ANCHOR_SCALES,
                            scale_factors=cfgs.SCALE_FACTORS,
                            base_anchor_size_list=cfgs.BASE_ANCHOR_SIZE_LIST,  # P2, P3, P4, P5, P6
                            level=cfgs.LEVEL,
                            top_k_nms=cfgs.RPN_TOP_K_NMS,
                            rpn_nms_iou_threshold=cfgs.RPN_NMS_IOU_THRESHOLD,
                            max_proposals_num=cfgs.MAX_PROPOSAL_NUM,
                            rpn_iou_positive_threshold=cfgs.RPN_IOU_POSITIVE_THRESHOLD,
                            rpn_iou_negative_threshold=cfgs.RPN_IOU_NEGATIVE_THRESHOLD,
                            rpn_mini_batch_size=cfgs.RPN_MINIBATCH_SIZE,
                            rpn_positives_ratio=cfgs.RPN_POSITIVE_RATE,
                            remove_outside_anchors=False,  # whether remove anchors outside
                            rpn_weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME])

        # rpn predict proposals
        rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals()  # rpn_score shape: [300, ]

        # ***********************************************************************************************
        # *                                         Fast RCNN                                           *
        # ***********************************************************************************************
        fast_rcnn = build_fast_rcnn.FastRCNN(feature_pyramid=rpn.feature_pyramid,
                                             rpn_proposals_boxes=rpn_proposals_boxes,
                                             rpn_proposals_scores=rpn_proposals_scores,
                                             img_shape=tf.shape(img_batch),
                                             img_batch=img_batch,
                                             roi_size=cfgs.ROI_SIZE,
                                             roi_pool_kernel_size=cfgs.ROI_POOL_KERNEL_SIZE,
                                             scale_factors=cfgs.SCALE_FACTORS,
                                             gtboxes_and_label=None,
                                             gtboxes_and_label_minAreaRectangle=gtboxes_and_label_minAreaRectangle,
                                             fast_rcnn_nms_iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                             fast_rcnn_maximum_boxes_per_img=100,
                                             fast_rcnn_nms_max_boxes_per_class=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                             show_detections_score_threshold=cfgs.FINAL_SCORE_THRESHOLD,
                                             # show detections which score >= 0.6
                                             num_classes=cfgs.CLASS_NUM,
                                             fast_rcnn_minibatch_size=cfgs.FAST_RCNN_MINIBATCH_SIZE,
                                             fast_rcnn_positives_ratio=cfgs.FAST_RCNN_POSITIVE_RATE,
                                             fast_rcnn_positives_iou_threshold=cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD,
                                             # iou>0.5 is positive, iou<0.5 is negative
                                             use_dropout=cfgs.USE_DROPOUT,
                                             weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME],
                                             is_training=False,
                                             level=cfgs.LEVEL,
                                             head_quadrant=head_quadrant)

        fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category, \
        fast_rcnn_decode_boxes_rotate, fast_rcnn_score_rotate, fast_rcnn_head_quadrant, \
        num_of_objects_rotate, detection_category_rotate = fast_rcnn.fast_rcnn_predict()

        # train
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        restorer, restore_ckpt = restore_model.get_restorer()

        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            if not restorer is None:
                restorer.restore(sess, restore_ckpt)
                print('restore model')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            gtboxes_horizontal_dict = {}
            gtboxes_rotate_dict = {}

            all_boxes_h = []
            all_boxes_r = []
            all_img_names = []

            for i in range(img_num):

                start = time.time()

                _img_name_batch, _img_batch, _gtboxes_and_label, _gtboxes_and_label_minAreaRectangle, \
                _fast_rcnn_decode_boxes, _fast_rcnn_score, _detection_category, _fast_rcnn_decode_boxes_rotate, \
                _fast_rcnn_score_rotate, _detection_category_rotate \
                    = sess.run([img_name_batch, img_batch, gtboxes_and_label, gtboxes_and_label_minAreaRectangle,
                                fast_rcnn_decode_boxes, fast_rcnn_score, detection_category, fast_rcnn_decode_boxes_rotate,
                                fast_rcnn_score_rotate, detection_category_rotate])
                end = time.time()

                # gtboxes convert dict
                gtboxes_horizontal_dict[str(_img_name_batch[0])] = []
                gtboxes_rotate_dict[str(_img_name_batch[0])] = []

                gtbox_horizontal_list, gtbox_rotate_list = make_dict_packle(_gtboxes_and_label, _gtboxes_and_label_minAreaRectangle)

                xmin, ymin, xmax, ymax = _fast_rcnn_decode_boxes[:, 1], _fast_rcnn_decode_boxes[:, 0], _fast_rcnn_decode_boxes[:, 3], _fast_rcnn_decode_boxes[:, 2]
                x_c, y_c, w, h, theta = _fast_rcnn_decode_boxes_rotate[:, 1], _fast_rcnn_decode_boxes_rotate[:, 0], _fast_rcnn_decode_boxes_rotate[:, 3], \
                                        _fast_rcnn_decode_boxes_rotate[:, 2], _fast_rcnn_decode_boxes_rotate[:, 4]
                boxes_h = np.transpose(np.stack([xmin, ymin, xmax, ymax]))
                boxes_r = np.transpose(np.stack([x_c, y_c, w, h, theta]))
                dets_h = np.hstack((_detection_category.reshape(-1,1), _fast_rcnn_score.reshape(-1,1), boxes_h))
                dets_r = np.hstack((_detection_category_rotate.reshape(-1,1), _fast_rcnn_score_rotate.reshape(-1,1), boxes_r))
                all_boxes_h.append(dets_h)
                all_boxes_r.append(dets_r)
                all_img_names.append(str(_img_name_batch[0]))

                gtboxes_horizontal_dict[str(_img_name_batch[0])].extend(gtbox_horizontal_list)
                gtboxes_rotate_dict[str(_img_name_batch[0])].extend(gtbox_rotate_list)

                print(str(_img_name_batch[0]))

                view_bar('{} image cost {}s'.format(str(_img_name_batch[0]), (end - start)), i + 1, img_num)

            write_voc_results_file(all_boxes_h, all_img_names, cfgs.EVALUATE_R_DIR, 0)
            write_voc_results_file(all_boxes_r, all_img_names, cfgs.EVALUATE_R_DIR, 1)

            fw1 = open('gtboxes_horizontal_dict.pkl', 'wb')
            fw2 = open('gtboxes_rotate_dict.pkl', 'wb')
            pickle.dump(gtboxes_horizontal_dict, fw1)
            pickle.dump(gtboxes_rotate_dict, fw2)
            fw1.close()
            fw2.close()
            coord.request_stop()
            coord.join(threads)


def write_voc_results_file(all_boxes, all_img_names, det_save_dir, mode):
  '''
  :param all_boxes: is a list. each item reprensent the detections of a img.
  the detections is a array. shape is [-1, 7]. [category, score, x, y, w, h, theta]
  Note that: if none detections in this img. that the detetions is : []
  :param all_img_names: a  list of containing all the names of predicted imgs
  :param det_save_path:
  :mode 0: horizontal, 1: rotated
  :return:
  '''
  for cls, cls_id in NAME_LABEL_MAP.items():
    if cls == 'back_ground':
      continue
    print("Writing {} VOC resutls file".format(cls))

    mkdir(det_save_dir)
    if mode == 0:
        det_save_path = os.path.join(det_save_dir, "det_h_"+cls+".txt")
    else:
        det_save_path = os.path.join(det_save_dir, "det_r_"+cls+".txt")
    with open(det_save_path, 'wt') as f:
      for index, img_name in enumerate(all_img_names):
        this_img_detections = all_boxes[index]

        this_cls_detections = this_img_detections[this_img_detections[:, 0] == cls_id]
        if this_cls_detections.shape[0] == 0:
          continue # this cls has none detections in this img
        for a_det in this_cls_detections:
            if mode == 0:
                # that is [img_name, score, xmin, ymin, xmax, ymax]
                f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(img_name, a_det[1], a_det[2], a_det[3], a_det[4], a_det[5]))
            else:
                # that is [img_name, score, x, y, w, h, theta]
                f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(img_name, a_det[1], a_det[2], a_det[3], a_det[4], a_det[5], a_det[6]))  


def do_python_eval(mode):
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    fr1 = open('gtboxes_horizontal_dict.pkl', 'rb')
    fr2 = open('gtboxes_rotate_dict.pkl', 'rb')
    gtboxes_horizontal_dict = pickle.load(fr1)
    gtboxes_rotate_dict = pickle.load(fr2)
    fr1.close()
    fr2.close()

    AP_list = []
    for cls, index in NAME_LABEL_MAP.items():
        if cls == 'back_ground':
            continue
        recall, precision, AP = voc_eval(detpath=cfgs.EVALUATE_R_DIR,
                                         gtboxes_horizontal_dict=gtboxes_horizontal_dict,
                                         gtboxes_rotate_dict=gtboxes_rotate_dict,
                                         cls_name=cls,
                                         mode=mode)
        AP_list += [AP]
        print("cls : {}|| Recall: {} || Precison: {}|| AP: {}".format(cls, recall[-1], precision[-1], AP))
        c = colors.cnames.keys()
        # c_dark = list(filter(lambda x: x.startswith('dark'), c)) # only supports up to 20 classes
        c_dark = list(filter(lambda x: x.startswith('d'), c)) # 24 classes
        # c = ['red', 'orange']
        plt.axis([0, 1.2, 0, 1])
        plt.plot(recall, precision, color=c_dark[index], label=cls)
    plt.legend(loc='upper right')
    plt.xlabel('R')
    plt.ylabel('P')
    if mode == 0:
        plt.savefig('./PR_R_h.png')
    else:
        plt.savefig('./PR_R_r.png')
    print("mAP is : {}".format(np.mean(AP_list)))


def voc_eval(detpath, gtboxes_horizontal_dict, gtboxes_rotate_dict, cls_name, mode, ovthresh=0.5,
                 use_07_metric=False):
    '''
    :param detpath:
    :param annopath:
    :param test_imgid_list: it 's a list that contains the img_name of test_imgs
    :param cls_name:
    :param ovthresh:
    :param use_07_metric:
    :param use_diff:
    :return:
    '''
    
    # get gtboxes for this class.
    # **************************************
    class_recs = {}
    num_pos = 0
    for imagename in gtboxes_horizontal_dict.keys():
        if mode == 0:
            R = [obj for obj in gtboxes_horizontal_dict[imagename] if obj['name'] == cls_name]
        else:
            R = [obj for obj in gtboxes_rotate_dict[imagename] if obj['name'] == cls_name]
        bbox = np.array([x['bbox'] for x in R])
        det = [False] * len(R)
        num_pos = num_pos + len(R)
        class_recs[imagename] = {'bbox': bbox, 'det': det} # det means that gtboxes has already been detected
    
    # read the detection file
    if mode == 0:
        detfile = os.path.join(detpath, "det_h_"+cls_name+".txt")
    else:
        detfile = os.path.join(detpath, "det_r_"+cls_name+".txt")
    with open(detfile, 'r') as f:
        lines = f.readlines()
    
    # for a line. that is [img_name, confidence, xmin, ymin, xmax, ymax]
    splitlines = [x.strip().split(' ') for x in lines]  # a list that include a list
    image_ids = [x[0] for x in splitlines]  # img_id is img_name
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    
    nd = len(image_ids) # num of detections. That, a line is a det_box.
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    
    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]  #reorder the img_name
        
        # go down dets and mark TPs and FPs
        for d in range(nd):
            R = class_recs[image_ids[d]]  # img_id is img_name
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                if mode == 0:
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih
                    
                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
                    
                    overlaps = inters / uni
                else:
                    overlaps = []
                    for i in range(len(BBGT)):
                        overlap = iou_rotate.iou_rotate_calculate1(np.array([bb]),
                                                                    np.array([BBGT[i]]),
                                                                    # BBGT[i].reshape(1,-1),
                                                                    use_gpu=False)[0]
                        overlaps.append(overlap)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            
            if ovmax > ovthresh:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.
    
    # 4. get recall, precison and AP
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(num_pos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    if len(tp) > 0 and len(fp) > 0:
      return rec, prec, ap
    else:
      return np.array([0], np.float64), np.array([0], np.float64), ap

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


if __name__ == '__main__':
    img_num = 571 # Need to equal to the number of test images, otherwise it will continuely get the image 
    eval_ship(img_num)
    # 0: horizontal standard 1: rotate standard
    print('**************Evaluating horizontal predictions:**************')
    do_python_eval(0)
    print('**************Evaluating rotated predictions:**************')
    do_python_eval(1)

