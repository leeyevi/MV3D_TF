from fast_rcnn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
from utils.cython_nms import nms, nms_new
from utils.boxes_grid import get_boxes_grid
from utils.transform import lidar_3d_to_corners, corners_to_bv, lidar_cnr_to_img_single, lidar_cnr_to_img
from utils.draw import show_lidar_corners, show_image_boxes
import cPickle
import heapq
from utils.blob import im_list_to_blob, lidar_list_to_blob
import os
import math
from rpn_msr.generate import imdb_proposals_det
import tensorflow as tf
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv, bbox_transform_inv_cnr
import matplotlib.pyplot as plt
from tensorflow.python.client import timeline
import time

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    # im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    #  for target_size in cfg.TEST.SCALES:
        #  im_scale = float(target_size) / float(im_size_min)
        #  # Prevent the biggest axis from being more than MAX_SIZE
        #  if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            #  im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        #  im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        #  interpolation=cv2.INTER_LINEAR)
        #  im_scale_factors.append(im_scale)
    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_lidar_bv_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    # im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    # im_scale_factors = []

    # for target_size in cfg.TEST.SCALES:
    #     im_scale = float(target_size) / float(im_size_min)
    #     # Prevent the biggest axis from being more than MAX_SIZE
    #     if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
    #         im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    #     # im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
    #     #                 interpolation=cv2.INTER_LINEAR)
    #     im_scale_factors.append(im_scale)
    processed_ims.append(im_orig)

    # Create a blob to hold the input images
    blob = lidar_list_to_blob(processed_ims)

    return blob, [1.0]

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob
    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob
    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)
    scales = np.array(scales)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im,bv, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'lidar_bv_data' : None, 'rois' : None}
    blobs['lidar_bv_data'], im_scale_factors = _get_lidar_bv_blob(bv)
    blobs['image_data'], _  = _get_image_blob(im)
    return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""

    for i in xrange(boxes.shape[0]):
        boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

    return boxes


def im_detect(sess, net, im, boxes=None):
    """Detect object classes in an image given object proposals.
    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
    # forward pass
    if cfg.TEST.HAS_RPN:
        feed_dict={net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}
    else:
        feed_dict={net.data: blobs['data'], net.rois: blobs['rois'], net.keep_prob: 1.0}

    run_options = None
    run_metadata = None
    if cfg.TEST.DEBUG_TIMELINE:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

    cls_score, cls_prob, bbox_pred, rois = sess.run([net.get_output('cls_score'), net.get_output('cls_prob'), net.get_output('bbox_pred'),net.get_output('rois')],
                                                    feed_dict=feed_dict,
                                                    options=run_options,
                                                    run_metadata=run_metadata)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        boxes = rois[:, 1:5] / im_scales[0]


    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = cls_score
    else:
        # use softmax estimated probabilities
        scores = cls_prob

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    if cfg.TEST.DEBUG_TIMELINE:
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        trace_file = open(str(long(time.time() * 1000)) + '-test-timeline.ctf.json', 'w')
        trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
        trace_file.close()

    return scores, pred_boxes

def bv_detect(sess, net, im, boxes=None):
    """Detect object classes in an lidar bv  given object proposals.
    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        bv (ndarray): lidar bv to test
        boxes (ndarray): R x 4 array of object proposals
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    blobs, im_scales = _get_blobs(im, boxes)


    im_blob = blobs['lidar_bv_data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)
    # forward pass
    if cfg.TEST.HAS_RPN:
        feed_dict={net.lidar_bv_data: blobs['lidar_bv_data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}

    run_options = None
    run_metadata = None
    if cfg.TEST.DEBUG_TIMELINE:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

    cls_score, cls_prob, bbox_pred_cnr, rois = sess.run([net.get_output('cls_score'),
                                             net.get_output('cls_prob'),
                                             net.get_output('bbox_pred'),
                                             net.get_output('rois')],
                                             feed_dict=feed_dict,
                                             options=run_options,
                                             run_metadata=run_metadata)

    scores = cls_prob

    print "scores: ", scores[:10]
    print "cls :", cls_score[:10]
    print "rois", rois[0].shape
    print "rois", np.where(rois[0][:,0] == 1)
    print "bbox_pred_cnr: ", bbox_pred_cnr[0]


    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        boxes_3d = rois[1][:, 1:7] / im_scales[0]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred_cnr
        boxes_cnr = lidar_to_corners(boxes_3d)
        boxes_cnr = np.hstack((boxes_cnr, boxes_cnr))
        pred_boxes_cnr = bbox_transform_inv_cnr(boxes_cnr, box_deltas)
        print "boxes_cnr: ", boxes_cnr[0]
        print "pred_boxes_cnr: ", pred_boxes_cnr[0]
        # pred_boxes = _clip_boxes(pred_boxes, im.shape)

    #  preject corners to lidar_bv
    pred_boxes_bv = corners_to_bv(pred_boxes_cnr)
    pred_boxes_bv = np.hstack((pred_boxes_bv, pred_boxes_bv))
    # pred_boxes_bv = rois[0]
    print pred_boxes_cnr.shape
    print pred_boxes_bv.shape


    if cfg.TEST.DEBUG_TIMELINE:
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        trace_file = open(str(long(time.time() * 1000)) + '-test-timeline.ctf.json', 'w')
        trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
        trace_file.close()

    return scores, pred_boxes_bv, pred_boxes_cnr

def box_detect(sess, net, im, bv, calib,  boxes=None):
    """Detect object classes in an lidar bv  given object proposals.
    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        bv (ndarray): lidar bv to test
        boxes (ndarray): R x 4 array of object proposals
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    blobs, im_scales = _get_blobs(im, bv, boxes)

    blobs['calib'] = calib
    bv_blob = blobs['lidar_bv_data']
    blobs['im_info'] = np.array(
        [[bv_blob.shape[1], bv_blob.shape[2], im_scales[0]]],
        dtype=np.float32)
    # forward pass
    feed_dict={net.lidar_bv_data: blobs['lidar_bv_data'],
               net.image_data: blobs['image_data'],
               net.im_info: blobs['im_info'],
               net.calib: blobs['calib'],
               net.keep_prob: 1.0}

    run_options = None
    run_metadata = None
    if cfg.TEST.DEBUG_TIMELINE:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

    cls_score, cls_prob, bbox_pred_cnr, rois = sess.run([net.get_output('cls_score'),
                                             net.get_output('cls_prob'),
                                             net.get_output('bbox_pred'),
                                             net.get_output('rois')],
                                             feed_dict=feed_dict,
                                             options=run_options,
                                             run_metadata=run_metadata)

    scores = cls_prob

    #  print "scores: ", scores[:10]
    #  print "cls :", cls_score[:10]
    #  print "rois", len(rois)
    #  print rois[1][:5]
    #  print "bbox_pred_cnr: ", bbox_pred_cnr[0]
    #  print ""


    assert len(im_scales) == 1, "Only single-image batch implemented"
    boxes_3d = rois[2][:, 1:7] / im_scales[0]

    # Apply bounding-box regression deltas
    box_deltas = bbox_pred_cnr
    #  print 'boxes_3d', boxes_3d
    boxes_cnr = lidar_3d_to_corners(boxes_3d)
    #  boxes_cnr = np.hstack((boxes_cnr, boxes_cnr))
    print boxes_cnr[0]
    pred_boxes_cnr = bbox_transform_inv_cnr(boxes_cnr, box_deltas)
    #  print "boxes_cnr: ", boxes_cnr[0]
    print "pred_boxes_cnr: ", pred_boxes_cnr[0]
    # pred_boxes = _clip_boxes(pred_boxes, im.shape)
    # print pred_boxes_cnr.shape

    #  preject corners to lidar_bv
    pred_boxes_bv = corners_to_bv(pred_boxes_cnr)
    #  pred_boxes_bv = np.hstack((pred_boxes_bv, pred_boxes_bv))
    #  pred_boxes_bv = rois[0]
    #  print pred_boxes_cnr.shape
    #  print pred_boxes_bv.shape
    #  pred_boxes_img = lidar_cnr_to_img(pred_boxes_cnr, calib[3], calib[2,:9], calib[0])

    # if cfg.TEST.DEBUG_TIMELINE:
    #     trace = timeline.Timeline(step_stats=run_metadata.step_stats)
    #     trace_file = open(str(long(time.time() * 1000)) + '-test-timeline.ctf.json', 'w')
    #     trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
    #     trace_file.close()

    #  return scores, pred_boxes_bv, pred_boxes_img, pred_boxes_cnr
    return scores, pred_boxes_bv, pred_boxes_cnr

def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    #im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            #plt.cla()
            #plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.gca().text(bbox[0], bbox[1] - 2,
                 '{:s} {:.3f}'.format(class_name, score),
                 bbox=dict(facecolor='blue', alpha=0.5),
                 fontsize=14, color='white')

            plt.title('{}  {:.3f}'.format(class_name, score))
    #plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue

            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            scores = dets[:, 4]
            inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.TEST.DET_THRESHOLD))[0]
            dets = dets[inds,:]
            if dets == []:
                continue

            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def test_net(sess, net, imdb, weights_filename , max_per_image=300, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    #    all_boxes_cnr[cls][image] = N x 25 array of detections in
    #    (x0-x7, y0-y7, z0-z7, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    all_boxes_img = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    all_boxes_cnr = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, weights_filename)
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    # with tf.variable_scope('deconv_2x_1'):
        # bar1 = tf.get_variable("bar", (2,3)) # create
    #  deconv1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv1_2")
    #  print deconv1[0].eval(session = sess)
    # deconv2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="deconv_4x_1")
    # sess.run(tf.variables_initializer([deconv1[0], deconv1[1], deconv2[0], deconv2[1]], name='init'))
    # print deconv1[0].eval(session = sess)[0]
    # conv5_3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv5_3")[0]
    # deconv1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="deconv_2x_1")[0]
    # deconv2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="deconv_4x_1")[0]
    # shape_conv5_3 = conv5_3.get_shape().as_list()
    # shape1 = deconv1.get_shape().as_list()
    # shape2 = deconv2.get_shape().as_list()
    # print 'conv5_3 shape', shape_conv5_3
    # print 'deconv_2x_1 shape', shape1
    # print 'deconv_4x_1 shape', shape2


    for i in xrange(num_images):
    # for i in xrange(1):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None

        # _t['im_detect'].tic()
        # scores, boxes = im_detect(sess, net, im, box_proposals)
        # _t['im_detect'].toc()

        im = cv2.imread(imdb.image_path_at(i))
        bv = np.load(imdb.lidar_path_at(i))
        calib = imdb.calib_at(i)

        print "Inference: ", imdb.lidar_path_at(i)
        # print np.where(bv != 0)

        _t['im_detect'].tic()
        scores, boxes_bv, boxes_cnr = box_detect(sess, net, im, bv, calib,  box_proposals)
        _t['im_detect'].toc()


        _t['misc'].tic()
        if vis:
            image = im[:, :, (2, 1, 0)]
            plt.cla()
            plt.imshow(image)

        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes_bv[inds, j*4:(j+1)*4]
            cls_boxes_cnr = boxes_cnr[inds, j*24:(j+1)*24]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            cls_dets_cnr = np.hstack((cls_boxes_cnr, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            print "scores: ", cls_scores[:10]
            # print "boxes_bv: ", boxes_bv
            # print " cls_boxes", cls_boxes
            # print cls_boxes_cnr[:10]
            print "cls_dets : ", cls_dets.shape
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            cls_dets_cnr = cls_dets_cnr[keep, :]
            # project to image
            # cls_des_img = lidar_cnr_to_img_single(cls_dets_cnr, calib[3], calib[2], calib[0])
            # if vis:
            if np.any(cls_dets_cnr):


                image_bv = show_image_boxes(bv[:,:,9], cls_dets[:, :4])
                image_cnr = show_lidar_corners(im, cls_dets_cnr[:,:24], calib)

                plt.title('proposal_layer ')

                plt.subplot(211)
                plt.imshow(image_bv)
                plt.subplot(212)
                plt.imshow(image_cnr)
                plt.show()
                # vis_detections(image, imdb.classes[j], cls_dets_img)
            all_boxes[j][i] = cls_dets
            # all_boxes_img[j][i] = cls_des_img
            all_boxes_cnr[j][i] = cls_dets_cnr
        if vis:
           plt.show()
        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
                    # all_boxes_img[j][i] = all_boxes_img[j][i][keep, :]
                    all_boxes_cnr[j][i] = all_boxes_cnr[j][i][keep, :]
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    det_cnr_file = os.path.join(output_dir, 'detections_cnr.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes_cnr, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, all_boxes_cnr, output_dir)

