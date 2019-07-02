# Ship-Detection
Ship Detection based on FPN, horizontal bounding box, rotated bounding box, and head prediction.

This work is based on Yang xue, etc ([Github](https://github.com/yangxue0827/R2CNN_HEAD_FPN_Tensorflow)). Thanks to their great work. Based on their work, I have made following modifications:

## Focal loss

You need to lower the "FINAL_SCORE_THRESHOLD" value (in the cfgs.py) to see more prediction results.

## Randomly ratotation of data augmentation

When each time we train a image (batch size = 1), we randomly rotate the image by a angle within [0,90]. The expriments reveal that it is an efficient way as regularization.

Attention: the loss of ship head regression is multiplied by 10 and added into the total loss.

## Correction of eval.py (eval_deprecated.py) to EVAL_TEST.py for evaluation

Since their original evaluation code (eval.py) is not recommended, I have upgraded this as EVAL_TEST.py, in which it includes following concerns:

### Inconsitent Coordinate

1. Format of coordinates of predicted bounding boxes and ground truth bounding boxes:
	fast_rcnn_decode_boxes/gtbox_minAreaRectangle: [ymin, xmin, ymax, xmax]
	fast_rcnn_decode_boxes_rotate/gtbox: [y_c, x_c, h, w, theta]
2. However, when caculating IoUs on horizontal and rotated bounding boxes (EVAL_test.py and iou_rotate.py), it requires following format:
	Rotated: [x_c, y_c, w, h, theta]
	Horizontal: [xmin, ymin, xmax, ymax]

### Calculating the interpolation performed in all points

In some images there are more than one detection overlapping a ground truth. For those cases the detection with the highest IOU is taken, discarding the other detections. This rule is applied by the PASCAL VOC 2012 metric. However, in my upgraded implementatioin ([Reference](https://github.com/DetectionTeamUCAS/R2CNN_Faster-RCNN_Tensorflow)), the detection with the highest condidence is taken. I assume it is a tradeoff between classification and regression.

### When evaluating, you need to set the number of images needed to be evaluated manually.

### Need to further verification

## Change to class-agnostic regression

Originally, regression and NMS are committed for each class. Due to limited size of training data, I change the multi-classes regression to class-agnostic regression. Especially, the scores used for NMS are selected according to the highest confidence in all classes corresponding to each instance (each row). However, the values of fast_rcnn_loc_loss and fast_rcnn_loc_rotate_loss are too small, and it may causes unnormal gradient behavior. Therefore, you need to lower the learning rate to continue the training process and further analyze such phenomenon according to related works and Tensorflow debugger.

## Dataset

[HRSC 2016](https://sites.google.com/site/hrsc2016/home)

## Reference

### [Metrics for object detection](https://github.com/MonsterZhZh/Object-Detection-Metrics)

### [Oriented Evaluation Server provided by DOTA](https://github.com/CAPTAIN-WHU/DOTA_devkit)

### [Tensorflow debugger(from tensorflow.python import debug as tf_debug)](https://blog.csdn.net/qq_22291287/article/details/82712050)
