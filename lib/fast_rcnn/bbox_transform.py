# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# from __future__ import absolute_division

import numpy as np

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_3d(ex_rois_3d, gt_rois_3d):

    # x, y, z, l, w, h
    ex_ctr_x = ex_rois_3d[:, 0]
    ex_ctr_y = ex_rois_3d[:, 1]
    ex_ctr_z = ex_rois_3d[:, 2]
    ex_lengths = ex_rois_3d[:, 3]
    ex_widths = ex_rois_3d[:, 4]
    ex_heights = ex_rois_3d[:, 5]

    gt_ctr_x = gt_rois_3d[:, 0]
    gt_ctr_y = gt_rois_3d[:, 1]
    gt_ctr_z = gt_rois_3d[:, 2]
    gt_lengths = gt_rois_3d[:, 3]
    gt_widths = gt_rois_3d[:, 4]
    gt_heights = gt_rois_3d[:, 5]

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_lengths
    targets_dz = (gt_ctr_z - ex_ctr_z) / ex_heights
    targets_dl = np.log(gt_lengths / ex_lengths)
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dz, targets_dl, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_cnr(ex_rois_3d, gt_rois_3d):

    gt_xyz0 = gt_rois_3d[:, 0::8]
    gt_xyz6 = gt_rois_3d[:, 5::8]

    mean_xyz0 = gt_xyz0.mean(0)
    mean_xyz6 = gt_xyz6.mean(0)

    assert(mean_xyz0.shape[0] == 3)
    assert(mean_xyz6.shape[0] == 3)
    # box diagonal distance
    diag = np.linalg.norm(mean_xyz0 - mean_xyz6)
    assert diag != 0, "diagonal distance could not be zero"

    targets = (gt_rois_3d[:,:] - ex_rois_3d[:,:]) / diag

    return targets


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def bbox_transform_inv_3d(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    # widths = boxes[:, 2] - boxes[:, 0] + 1.0
    # heights = boxes[:, 3] - boxes[:, 1] + 1.0
    # ctr_x = boxes[:, 0] + 0.5 * widths
    # ctr_y = boxes[:, 1] + 0.5 * heights
    # print("boxes shape", boxes.shape)

    lengths = boxes[:, 3]
    widths = boxes[:, 4]
    heights = boxes[:, 5]
    ctr_x = boxes[:, 0]
    ctr_y = boxes[:, 1]
    ctr_z = boxes[:, 2]

    dx = deltas[:, 0::6] # stride = 6
    dy = deltas[:, 1::6]
    dz = deltas[:, 2::6]
    dl = deltas[:, 3::6]
    dw = deltas[:, 4::6]
    dh = deltas[:, 5::6]

    pred_ctr_x = dx * lengths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * widths[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_ctr_z = dz * heights[:, np.newaxis] + ctr_z[:, np.newaxis]
    pred_l = np.exp(dl) * lengths[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x
    pred_boxes[:, 0::6] = pred_ctr_x
    # y
    pred_boxes[:, 1::6] = pred_ctr_y
    # z
    pred_boxes[:, 2::6] = pred_ctr_z
    # l
    pred_boxes[:, 3::6] = pred_l
    # w
    pred_boxes[:, 4::6] = pred_w
    # h
    pred_boxes[:, 5::6] = pred_h

    return pred_boxes

def bbox_transform_inv_cnr(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    gt_xyz0 = boxes[:, 0::8]
    gt_xyz6 = boxes[:, 5::8]

    mean_xyz0 = gt_xyz0.mean(0)
    mean_xyz6 = gt_xyz6.mean(0)

    # box diagonal distance
    diag = np.linalg.norm(mean_xyz0[:3] - mean_xyz6[:3])

    deltas = deltas * diag
    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)

    for i in range(deltas.shape[1]/24):
        pred_boxes[:,(i*24):(i*24+24)] = deltas[:,(i*24):(i*24+24)] + boxes

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
