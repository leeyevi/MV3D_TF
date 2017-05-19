# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob


def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # print("num_images: ", num_images)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                     size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    # im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    im_scales = [1]
    im = cv2.imread(roidb[0]['image_path'])
    im = im.astype(np.float32, copy=False)
    lidar_bv_blob = np.load(roidb[0]['lidar_bv_path'])

    # substract mean
    im -= cfg.PIXEL_MEANS

    im_blob = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
    lidar_bv_blob = lidar_bv_blob.reshape((1, lidar_bv_blob.shape[0], lidar_bv_blob.shape[1], lidar_bv_blob.shape[2]))

    blobs = {'image_data': im_blob,
             'lidar_bv_data': lidar_bv_blob}

    blobs['calib'] = roidb[0]['calib']

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"
    # gt boxes: (x1, y1, x2, y2, cls)
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    # gt boxes bv: (x1, y1, x2, y2, cls)
    gt_boxes_bv = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes_bv[:, 0:4] = roidb[0]['boxes_bv'][gt_inds, :]
    gt_boxes_bv[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes_bv'] = gt_boxes_bv

    # gt boxes 3d: (x, y, z, l, w, h, cls)
    gt_boxes_3d = np.empty((len(gt_inds), 7), dtype=np.float32)
    gt_boxes_3d[:, 0:6] = roidb[0]['boxes_3D'][gt_inds, :]
    gt_boxes_3d[:, 6] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes_3d'] = gt_boxes_3d
    # gt boxes corners: (x0, ... x7, y0, y1, ... y7, z0, ... z7, cls)
    gt_boxes_corners = np.empty((len(gt_inds), 25), dtype=np.float32)
    gt_boxes_corners[:, 0:24] = roidb[0]['boxes_corners'][gt_inds, :]
    gt_boxes_corners[:, 24] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes_corners'] = gt_boxes_corners

    blobs['im_info'] = np.array(
        [[lidar_bv_blob.shape[1], lidar_bv_blob.shape[2], im_scales[0]]],
        dtype=np.float32)

    return blobs



def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image_path'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

# def _project_im_rois(im_rois, im_scale_factor):
#     """Project image RoIs into the rescaled training image."""
#     rois = im_rois * im_scale_factor
#     return rois

# def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
#     """Visualize a mini-batch for debugging."""
#     import matplotlib.pyplot as plt
#     for i in xrange(rois_blob.shape[0]):
#         rois = rois_blob[i, :]
#         im_ind = rois[0]
#         roi = rois[1:]
#         im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
#         im += cfg.PIXEL_MEANS
#         im = im[:, :, (2, 1, 0)]
#         im = im.astype(np.uint8)
#         cls = labels_blob[i]
#         plt.imshow(im)
#         print 'class: ', cls, ' overlap: ', overlaps[i]
#         plt.gca().add_patch(
#             plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
#                           roi[3] - roi[1], fill=False,
#                           edgecolor='r', linewidth=3)
#             )
#         plt.show()
