# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors_bv, generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform, bbox_transform_3d
from utils.transform import bv_anchor_to_lidar
import pdb

DEBUG = True

def anchor_target_layer(rpn_cls_score, gt_boxes, gt_boxes_3d, im_info, _feat_stride = [16,], anchor_scales = [8, 16, 32]):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    _anchors = generate_anchors_bv()
    #  _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]

    if DEBUG:
        print 'anchors:'
        print _anchors.shape
        print 'anchor shapes:'
        print np.hstack((
            _anchors[:, 2::4] - _anchors[:, 0::4],
            _anchors[:, 3::4] - _anchors[:, 1::4],
        ))
        _counts = cfg.EPS
        _sums = np.zeros((1, 6))
        _squared_sums = np.zeros((1, 6))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0

    # allow boxes to sit over the edge by a small amount
    _allowed_border =  0
    # map of shape (..., H, W)
    #height, width = rpn_cls_score.shape[1:3]

    im_info = im_info[0]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap

    assert rpn_cls_score.shape[0] == 1, \
        'Only single item batches are supported'

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]

    if DEBUG:
        print 'AnchorTargetLayer: height', height, 'width', width
        print ''
        print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
        print 'scale: {}'.format(im_info[2])
        print 'height, width: ({}, {})'.format(height, width)
        print 'rpn: gt_boxes.shape', gt_boxes.shape
        print 'rpn: gt_boxes', gt_boxes
        print 'feat_stride', _feat_stride

    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]

    if DEBUG:
        print 'total_anchors: ', total_anchors
        print 'inds_inside: ', len(inds_inside)

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    if DEBUG:
        print 'anchors.shape: ', anchors.shape

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        # print max_overlaps.shape
        # print overlaps.shape
        # print 'argmax : ', np.where(np.logical_and(0 < max_overlaps, max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP) != False)
        # print max_overlaps[np.where(np.logical_and(0 < max_overlaps, max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP) != False)]
        # print 'argmax:', np.where(((max_overlaps > 0) & np.where(max_overlaps<0.5)))
        # labels[ (max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP) & ( 0 < max_overlaps)] = 0

        # hard negative for proposal_target_layer
        hard_negative = np.logical_and(0 < max_overlaps, max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP)
        labels[hard_negative] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1

    # random sample 

    # fg label: above threshold IOU
    # print np.where(max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP)
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1


        #print "was %s inds, disabling %s, now %s inds" % (
            #len(bg_inds), len(disable_inds), np.sum(labels == 0))

    # idx_label  = np.where(labels != -1)[0]
    # idx_target = np.where(labels ==  1)[0]
    # inds   = inds_inside[idx_label]
    # labels = labels[idx_label]

    # pos_anchors = anchors[idx_target]

    # pos_inds = inds_inside[idx_target]
    # inside_anchors = anchors[inds_inside]
    # pos_gt_boxes_3d = (gt_boxes_3d[argmax_overlaps])[idx_target]
    # if DEBUG:
    #     print 'pos_gt_boxes_3d shape: ', pos_gt_boxes_3d.shape
        #  print 'anchors shape', anchors.shape
    #  bbox_targets = np.zeros((len(inds_inside), 6), dtype=np.float32)
    #  anchors_3d = bv_anchor_to_lidar(anchors)
    #  bbox_targets = _compute_targets_3d(anchors_3d, gt_boxes_3d[argmax_overlaps, :])


    # bbox_targets = np.zeros((len(idx_target), 6), dtype=np.float32)
    # bbox_targets = _compute_targets_3d(anchors, gt_boxes_3d[argmax_overlaps, :])
    # anchors_3d = bv_anchor_to_lidar(pos_anchors)
    # bbox_targets = _compute_targets_3d(anchors_3d, pos_gt_boxes_3d)
    anchors_3d = bv_anchor_to_lidar(anchors)
    bbox_targets = _compute_targets_3d(anchors_3d, gt_boxes_3d[argmax_overlaps, :])


    # print 'labels = 0:, ', np.where(labels == 0)
    all_inds = np.where(labels != -1)
    labels_new = labels[all_inds]
    zeros = np.zeros((labels_new.shape[0], 1), dtype=np.float32)
    anchors =  np.hstack((zeros, anchors[all_inds])).astype(np.float32)
    anchors_3d =  np.hstack((zeros, anchors_3d[all_inds])).astype(np.float32)


    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
    # labels[hard_negative] = -1
    # # subsample negative labels if we have too many
    # num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    # bg_inds = np.where(labels != 1)[0]
    # # print len(bg_inds)
    # if len(bg_inds) > num_bg:
    #     disable_inds = npr.choice(
    #         bg_inds, size=(num_bg), replace=False)
    #     labels[disable_inds] = 0

    # all_inds = np.where(labels != -1)
    # labels_new = labels[all_inds]
    # zeros = np.zeros((labels_new.shape[0], 1), dtype=np.float32)
    # # print zeros.shape
    # # print len(all_inds)
    # anchors =  np.hstack((zeros, anchors[all_inds])).astype(np.float32)
    # anchors_3d =  np.hstack((zeros, anchors_3d[all_inds])).astype(np.float32)


    # bg_inds = np.where(hard_negative == True)[0]
    # disable_inds = npr.choice(
    #         bg_inds, size=(len(bg_inds)/2.), replace=False)
    # labels[disable_inds] = -1


    if DEBUG:
        _sums += bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means:'
        print means
        print 'stdevs:'
        print stds

    if DEBUG:
        print 'gt_boxes_3d: ', gt_boxes_3d[argmax_overlaps, :].shape
        print 'labels shape before unmap: ', labels.shape
        print 'targets shaoe before unmap: ', bbox_targets.shape
    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)

    if DEBUG:
        print 'rpn: max max_overlap', np.max(max_overlaps)
        print 'rpn: num_positive', np.sum(labels == 1)
        print 'rpn: num_negative', np.sum(labels == 0)
        _fg_sum += np.sum(labels == 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count
        #  print 'bbox targets: ', bbox_targets[fg_inds]
        print 'fg inds: ', fg_inds
        print 'label shape', labels.shape
        print 'bbox_targets', bbox_targets.shape
        #  print 'bbox 3d gt: ', gt_boxes_3d

    # labels
    #pdb.set_trace()
     # labels = labels[fg_inds]
    # labels = labels.reshape((1, height, width, A))
    #  labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    #  labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets
     # bbox_targets = bbox_targets[fg_inds]
    #  bbox_targets = bbox_targets \
        #  .reshape((1, height, width, A * 6)).transpose(0, 3, 1, 2)

    # bbox_targets = bbox_targets \
    #     .reshape((1, height, width, A * 6))

    rpn_bbox_targets = bbox_targets

    if DEBUG:
        print 'labels shape: ', labels.shape
        print 'targets shape: ', bbox_targets.shape


    return rpn_labels, rpn_bbox_targets, anchors, anchors_3d



def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

def _compute_targets_3d(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 6
    assert gt_rois.shape[1] == 7

    return bbox_transform_3d(ex_rois, gt_rois[:, :6]).astype(np.float32, copy=False)
