# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import yaml
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors_bv, generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_3d
from fast_rcnn.nms_wrapper import nms
from utils.transform import bv_anchor_to_lidar, lidar_to_bv, lidar_3d_to_bv, lidar_3d_to_corners, lidar_cnr_to_img
import pdb


DEBUG = False
"""
Outputs object detection proposals by applying estimated bounding-box
transformations to a set of regular boxes (called "anchors").
"""


# input
#  rpn_bbox_pred (nx6)
# return
# 1. rois: lidar_bv
# 4. rois_3d
def proposal_layer_3d(rpn_cls_prob_reshape,rpn_bbox_pred,im_info,calib,cfg_key, _feat_stride = [8,], anchor_scales=[1.0, 1.0]):
    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)

    #layer_params = yaml.load(self.param_str_)

    _anchors = generate_anchors_bv()
    #  _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]


    im_info = im_info[0]

    assert rpn_cls_prob_reshape.shape[0] == 1, \
        'Only single item batches are supported'
    # cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
    # cfg_key = 'TRAIN'
    pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
    min_size      = cfg[cfg_key].RPN_MIN_SIZE

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    # print rpn_cls_prob_reshape.shape

    height, width = rpn_cls_prob_reshape.shape[1:3]
    # scores = rpn_cls_prob_reshape[:, _num_anchors:, :, :]
    scores = np.reshape(np.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, 2])[:,:,:,:,1],[1, height, width, _num_anchors])

    bbox_deltas = rpn_bbox_pred


    if DEBUG:
        print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
        print 'scale: {}'.format(im_info[2])

    # 1. Generate proposals from bbox deltas and shifted anchors
    

    if DEBUG:
        print 'score map size: {}'.format(scores.shape)

    # Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    anchors = _anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    # bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 6))
    bbox_deltas = bbox_deltas.reshape((-1, 6))

    # Same story for the scores:
    #
    # scores are (1, A, H, W) format
    # transpose to (1, H, W, A)
    # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
    # scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
    scores = scores.reshape((-1, 1))

    # print np.sort(scores.ravel())[-30:]

    # convert anchors bv to anchors_3d
    anchors_3d = bv_anchor_to_lidar(anchors)
    # Convert anchors into proposals via bbox transformations
    proposals_3d = bbox_transform_inv_3d(anchors_3d, bbox_deltas)
    # convert back to lidar_bv
    proposals_bv = lidar_3d_to_bv(proposals_3d)

    lidar_corners = lidar_3d_to_corners(proposals_3d)
    proposals_img = lidar_cnr_to_img(lidar_corners,
                                calib[3], calib[2], calib[0])


    if DEBUG:
        # print "bbox_deltas: ", bbox_deltas[:10]
        # print "proposals number: ", proposals_3d[:10]
        print "proposals_bv shape: ", proposals_bv.shape
        print "proposals_3d shape: ", proposals_3d.shape

    # 2. clip predicted boxes to image
    proposals_bv = clip_boxes(proposals_bv, im_info[:2])

    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    keep = _filter_boxes(proposals_bv, min_size * im_info[2])
    proposals_bv = proposals_bv[keep, :]
    proposals_3d = proposals_3d[keep, :]
    proposals_img = proposals_img[keep, :]
    scores = scores[keep]

    # TODO: pass real image_info
    keep = _filter_img_boxes(proposals_img, [375, 1242])
    proposals_bv = proposals_bv[keep, :]
    proposals_3d = proposals_3d[keep, :]
    proposals_img = proposals_img[keep, :]
    scores = scores[keep]


    if DEBUG:
        print "proposals after clip"
        print "proposals_bv shape: ", proposals_bv.shape
        print "proposals_3d shape: ", proposals_3d.shape
        print "proposals_img shape: ", proposals_img.shape
    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals_bv = proposals_bv[order, :]
    proposals_3d = proposals_3d[order, :]
    proposals_img = proposals_img[order, :]
    scores = scores[order]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    keep = nms(np.hstack((proposals_bv, scores)), nms_thresh)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals_bv = proposals_bv[keep, :]
    proposals_3d = proposals_3d[keep, :]
    proposals_img = proposals_img[keep, :]
    scores = scores[keep]

    if DEBUG:
        print "proposals after nms"
        print "proposals_bv shape: ", proposals_bv.shape
        print "proposals_3d shape: ", proposals_3d.shape
        # print "proposals_bv", proposals_bv[:10]
        # print "proposals_img", proposals_img[:10]
        # print "scores: ", scores[:10]
    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    batch_inds = np.zeros((proposals_bv.shape[0], 1), dtype=np.float32)
    blob_bv = np.hstack((batch_inds, proposals_bv.astype(np.float32, copy=False)))
    blob_img = np.hstack((batch_inds, proposals_img.astype(np.float32, copy=False)))
    blob_3d = np.hstack((batch_inds, proposals_3d.astype(np.float32, copy=False)))

    if DEBUG:
        print "blob shape ====================:"
        print blob_bv.shape
        print blob_img.shape
        # print '3d', blob_3d[:10]
        # print lidar_corners[:10]
        # print 'bv', blob_bv[:10]
        # print 'img', blob_img[:10]

    return blob_bv, blob_img, blob_3d


def proposal_layer(rpn_cls_prob_reshape,rpn_bbox_pred,im_info,cfg_key,_feat_stride = [16,],anchor_scales = [8, 16, 32]):
    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)

    #layer_params = yaml.load(self.param_str_)
    # _anchors = generate_anchors_bv()
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]
    rpn_cls_prob_reshape = np.transpose(rpn_cls_prob_reshape,[0,3,1,2])
    rpn_bbox_pred = np.transpose(rpn_bbox_pred,[0,3,1,2])
    #rpn_cls_prob_reshape = np.transpose(np.reshape(rpn_cls_prob_reshape,[1,rpn_cls_prob_reshape.shape[0],rpn_cls_prob_reshape.shape[1],rpn_cls_prob_reshape.shape[2]]),[0,3,2,1])
    #rpn_bbox_pred = np.transpose(rpn_bbox_pred,[0,3,2,1])
    im_info = im_info[0]

    assert rpn_cls_prob_reshape.shape[0] == 1, \
        'Only single item batches are supported'
    # cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
    #cfg_key = 'TEST'
    pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
    min_size      = cfg[cfg_key].RPN_MIN_SIZE

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    scores = rpn_cls_prob_reshape[:, _num_anchors:, :, :]
    bbox_deltas = rpn_bbox_pred
    #im_info = bottom[2].data[0, :]

    if DEBUG:
        print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
        print 'scale: {}'.format(im_info[2])

    # 1. Generate proposals from bbox deltas and shifted anchors
    height, width = scores.shape[-2:]

    if DEBUG:
        print 'score map size: {}'.format(scores.shape)

    # Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    anchors = _anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

    # Same story for the scores:
    #
    # scores are (1, A, H, W) format
    # transpose to (1, H, W, A)
    # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

    anchors_3d = bv_anchor_to_lidar(anchors)

    # Convert anchors into proposals via bbox transformations
    proposals = bbox_transform_inv_3d(anchors_3d, bbox_deltas)

    # 2. clip predicted boxes to image
    proposals = clip_boxes(proposals, im_info[:2])

    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    keep = _filter_boxes(proposals, min_size * im_info[2])
    proposals = proposals[keep, :]
    scores = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    keep = nms(np.hstack((proposals, scores)), nms_thresh)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]
    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
    return blob
    #top[0].reshape(*(blob.shape))
    #top[0].data[...] = blob

    # [Optional] output scores blob
    #if len(top) > 1:
    #    top[1].reshape(*(scores.shape))
    #    top[1].data[...] = scores




def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

def _filter_img_boxes(boxes, im_info):
    """Remove all boxes with any side smaller than min_size."""
    padding = 50
    w_min = -padding
    w_max = im_info[1] + padding
    h_min = -padding
    h_max = im_info[0] + padding
    keep = np.where((w_min <= boxes[:,0]) & (boxes[:,2] <= w_max) & (h_min <= boxes[:,1]) &
                    (boxes[:,3] <= h_max))[0]
    return keep
