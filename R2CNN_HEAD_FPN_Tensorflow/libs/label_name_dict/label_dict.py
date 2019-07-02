# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from libs.configs import cfgs

if cfgs.DATASET_NAME == 'HRSC2016':
    # NAME_LABEL_MAP = {
    #     'back_ground': 0,
    #     'ship': 1
    # }
    # Class_ID distribution on the HRSC_2016 Train set:
        # 100000001:386
        # 100000002:1
        # 100000003:44
        # 100000004:1

        # 100000005:50
        # 100000006:20
        # 100000007:266
        # 100000008:66
        # 100000009:132
        # 100000010:30
        # 100000011:160
        # Without 100000012
        # 100000013:8
        # Without 100000014
        # 100000015:51
        # 100000016:99
        # Without 100000017
        # 100000018:34
        # 100000019:47
        # 100000020:15
        # 100000022:55
        # 100000024:8
        # 100000025:167
        # 100000026:6
        # 100000027:44
        # 100000028:5
        # 100000029:15
        # 100000030:25
        # Without 100000031
        # 100000032:13
        # Without 100000033

        # Total Instances: 1748
        # Total classes: 26
    NAME_LABEL_MAP = {
        'back_ground': 0,
        '100000005': 1,
        '100000006' : 2,
        '100000007' : 3,
        '100000008' : 4,
        '100000009' : 5,
        '100000010' : 6,
        '100000011' : 7,
        # 'Kitty' : 8,
        '100000013' : 8,
        # 'Abukuma' : 10,
        '100000015' : 9,
        '100000016' : 10,
        # 'USS' : 13,
        '100000018' : 11,
        '100000019' : 12,
        '100000020' : 13,
        '100000022' : 14,
        '100000024' : 15,
        '100000025' : 16,
        '100000026' : 17,
        '100000027' : 18,
        '100000028' : 19,
        '100000029' : 20,
        '100000030' : 21,
        # 'Ford_class' : 25,
        '100000032' : 22,
        # 'Invincible_class' : 27
    }
elif cfgs.DATASET_NAME == 'UAV':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'M41': 1,
        'M603A': 2,
        'M48H': 3,
    }
elif cfgs.DATASET_NAME == 'airplane':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'airplane': 1
    }
elif cfgs.DATASET_NAME == 'pascal':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    }
else:
    assert 'please set label dict!'


def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

LABEl_NAME_MAP = get_label_name_map()