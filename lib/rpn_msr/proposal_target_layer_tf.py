# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform_3d
from utils.cython_bbox import bbox_overlaps
import pdb

DEBUG = False

TOP_X_MAX = 70.3
TOP_X_MIN = 0
TOP_Y_MIN = -40
TOP_Y_MAX = 40
RES = 0.1
LIDAR_HEIGHT = 1.73
CAR_HEIGHT = 1.56

_count = 0

def proposal_target_layer_3d(rpn_rois_bv, gt_boxes_bv, gt_boxes_3d, _num_classes):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    
    if DEBUG:
        print("rpn rois shape: ", rpn_rois_bv.shape, rpn_rois_bv.dtype)

    # TODO(rbg): it's annoying that sometimes I have extra info before
    # and other times after box coordinates -- normalize to one format
    all_rois = rpn_rois_bv
    # Include ground-truth boxes in the set of candidate rois
    zeros = np.zeros((gt_boxes_bv.shape[0], 1), dtype=gt_boxes_bv.dtype)
    all_rois = np.vstack(
        (all_rois, np.hstack((zeros, gt_boxes_bv[:, :-1])))
    )

    # Sanity check: single batch only
    assert np.all(all_rois[:, 0] == 0), \
            'Only single item batches are supported'

    num_images = 1
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Sample rois with classification labels and bounding box regression
    # targets
    labels, rois, bbox_targets, bbox_inside_weights = _sample_rois_3d(
        all_rois, gt_boxes_bv, gt_boxes_3d, fg_rois_per_image,
        rois_per_image, _num_classes)

    if DEBUG:
        print 'num fg: {}'.format((labels > 0).sum())
        print 'num bg: {}'.format((labels == 0).sum())
        # _count += 1
        # _fg_num += (labels > 0).sum()
        # _bg_num += (labels == 0).sum()
        # print 'num fg avg: {}'.format(_fg_num / _count)
        # print 'num bg avg: {}'.format(_bg_num / _count)
        # print 'ratio: {:.3f}'.format(float(_fg_num) / float(_bg_num))

    rois = rois.reshape(-1,7).astype(np.float32)
    labels = labels.reshape(-1,1).astype(np.int32)
    bbox_targets = bbox_targets.reshape(-1,_num_classes*6).astype(np.float32)
    bbox_inside_weights = bbox_inside_weights.reshape(-1,_num_classes*6).astype(np.float32)

    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    return rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights


def proposal_target_layer(rpn_rois, gt_boxes,_num_classes):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    all_rois = rpn_rois
    # TODO(rbg): it's annoying that sometimes I have extra info before
    # and other times after box coordinates -- normalize to one format

    # Include ground-truth boxes in the set of candidate rois
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    all_rois = np.vstack(
        (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
    )

    # Sanity check: single batch only
    assert np.all(all_rois[:, 0] == 0), \
            'Only single item batches are supported'

    num_images = 1
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Sample rois with classification labels and bounding box regression
    # targets
    labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, gt_boxes, fg_rois_per_image,
        rois_per_image, _num_classes)

    if DEBUG:
        print 'num fg: {}'.format((labels > 0).sum())
        print 'num bg: {}'.format((labels == 0).sum())
        # _count += 1
        # _fg_num += (labels > 0).sum()
        # _bg_num += (labels == 0).sum()
        # print 'num fg avg: {}'.format(_fg_num / _count)
        # print 'num bg avg: {}'.format(_bg_num / _count)
        # print 'ratio: {:.3f}'.format(float(_fg_num) / float(_bg_num))

    rois = rois.reshape(-1,5)
    labels = labels.reshape(-1,1)
    bbox_targets = bbox_targets.reshape(-1,_num_classes*4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1,_num_classes*4)

    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    return rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = np.array(bbox_target_data[:, 0], dtype=np.uint16, copy=True)
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _get_bbox_regression_labels_3d(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = np.array(bbox_target_data[:, 0], dtype=np.uint16, copy=True)
    bbox_targets = np.zeros((clss.size, 6 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 6 * cls
        end = start + 6
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _compute_targets_3d(ex_rois_3d, gt_rois_3d, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois_3d.shape[0] == gt_rois_3d.shape[0]
    assert ex_rois_3d.shape[1] == 6
    assert gt_rois_3d.shape[1] == 6

    targets = bbox_transform_3d(ex_rois_3d, gt_rois_3d)
    # if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    #     # Optionally normalize targets by a precomputed mean and stdev
    #     targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
    #             / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois_3d(all_rois, gt_boxes_bv, gt_boxes_3d, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)

    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes_bv[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes_bv[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0

    # convert bird view rois to lidar coordinates
    all_rois_3d = _bv_roi_to_3d(all_rois)
    rois = all_rois_3d[keep_inds]

    bbox_target_data = _compute_targets_3d(
        rois[:, 1:7], gt_boxes_3d[gt_assignment[keep_inds], :6], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels_3d(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights

def _bv_to_lidar(x, y):
    Y0, Yn = 0, int((TOP_X_MAX-TOP_X_MIN)//RES)+1
    X0, Xn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//RES)+1
    yy = (Yn - y)*RES + TOP_Y_MIN
    xx = (Xn - x)*RES + TOP_X_MIN
    return xx, yy

def _lidar_to_bv(x,y,z=None):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//RES)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//RES)+1

    xx = Yn-int((y-TOP_Y_MIN)//RES)
    yy = Xn-int((x-TOP_X_MIN)//RES)

    return (xx,yy)


def _bv_roi_to_3d(rpn_rois_bv):
    """ convert rpn rois (0, x1, y1, x2, y2) to lidar 3d anchor (0, x, y, z, l, w, h) """
     # convert bird view rpn_rois to lidar coordinates
    rpn_rois_bv[:,1], rpn_rois_bv[:,2] = _bv_to_lidar(rpn_rois_bv[:,1], rpn_rois_bv[:,2])
    rpn_rois_bv[:,3], rpn_rois_bv[:,4] = _bv_to_lidar(rpn_rois_bv[:,3], rpn_rois_bv[:,4])

    # convert rpn_rois(0, x1, y1, x2, y2) to rpn_rois_ctr (0, x, y, l, w)
    rpn_rois_ctr = np.zeros(shape=rpn_rois_bv.shape, dtype=rpn_rois_bv.dtype)

    rpn_rois_ctr[:,1] = (rpn_rois_bv[:,1] + rpn_rois_bv[:,3]) * 0.5 # x
    rpn_rois_ctr[:,2] = (rpn_rois_bv[:,2] + rpn_rois_bv[:,4]) * 0.5 # y
    rpn_rois_ctr[:,3] = np.abs(rpn_rois_bv[:,3] - rpn_rois_bv[:,1]) # l
    rpn_rois_ctr[:,4] = np.abs(rpn_rois_bv[:,2] - rpn_rois_bv[:,4]) # w

    # extend rpn_rois_ctr (0, x, y) to 3d rois (0, x, y, z, l, w, h)
    rshape = rpn_rois_ctr.shape
    ctr_height = np.ones((rshape[0]))*(- (LIDAR_HEIGHT - CAR_HEIGHT * 0.5))
    car_height = np.ones((rshape[0]))*CAR_HEIGHT
    ctr_height = ctr_height.reshape(-1, 1)
    car_height = car_height.reshape(-1, 1)

    if DEBUG:
        print rpn_rois_ctr.shape
        print 'car height shape', car_height.shape
        print 'ctr shape', ctr_height.shape


    all_rois_3d = np.hstack((rpn_rois_ctr[:,:3],
                ctr_height, rpn_rois_ctr[:,3:5], car_height))
    assert(all_rois_3d.shape[1] == 7)
    return all_rois_3d

def _lidar_to_bv(rois_3d):

    rois = np.zeros((rois_3d.shape[0],5))
    rois[:, 0] = rois_3d[:, 0]

    rois[:, 1] = rois_3d[:, 1] + rois_3d[:, 4] * 0.5
    rois[:, 2] = rois_3d[:, 2] + rois_3d[:, 5] * 0.5
    rois[:, 3] = rois_3d[:, 1] - rois_3d[:, 4] * 0.5
    rois[:, 4] = rois_3d[:, 2] - rois_3d[:, 5] * 0.5

    rois[:, 1], rois[:, 2] = _lidar_to_bv(rois[:, 1], rois[:, 2])
    rois[:, 3], rois[:, 4] = _lidar_to_bv(rois[:, 3], rois[:, 4])              

    return rois

