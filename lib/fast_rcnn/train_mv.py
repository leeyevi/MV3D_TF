# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from fast_rcnn.config import cfg
import gt_data_layer.roidb as gdl_roidb
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.python.client import timeline
import time

# DEBUG = True
DEBUG = False
vis = True
# vis = False

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, saver, network, imdb, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        # print 'Computing bounding-box regression targets...'
        # if cfg.TRAIN.BBOX_REG:
        #     self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        # print 'done'

        # For checkpoint
        self.saver = saver

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

    def _modified_smooth_l1(self, sigma, bbox_pred, bbox_targets):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma

        diffs = tf.subtract(bbox_pred, bbox_targets)

        smooth_l1_sign = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(diffs, diffs), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(diffs), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))
        outside_mul = smooth_l1_result

        return outside_mul


    def train_model(self, sess, max_iters):
        """Network training loop."""

        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)

        # RPN
        # classification loss
        rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score_reshape'),[-1,2])
        rpn_label = tf.reshape(self.net.get_output('rpn_data')[0],[-1])

        rpn_keep = tf.where(tf.not_equal(rpn_label,-1))
        # only regression positive anchors
        rpn_bbox_keep = tf.where(tf.equal(rpn_label, 1))

        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_keep),[-1,2])
        # rpn_cls_score = tf.reshape(rpn_cls_score, [-1,2])
        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_keep),[-1])
        rpn_label = tf.reshape(rpn_label, [-1])
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

        # bounding box regression L1 loss
        rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
        #  rpn_bbox_targets = tf.transpose(self.net.get_output('rpn_data')[1],[0,2,3,1])
        rpn_bbox_targets = self.net.get_output('rpn_data')[1]




        rpn_bbox_pred = tf.reshape(tf.gather(tf.reshape(rpn_bbox_pred, [-1, 6]), rpn_bbox_keep),[-1, 6]) #
        rpn_bbox_targets = tf.reshape(tf.gather(tf.reshape(rpn_bbox_targets, [-1,6]),rpn_bbox_keep), [-1, 6])

        # rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 6])
        # rpn_bbox_targets = tf.reshape(rpn_bbox_targets, [-1,6])

        rpn_smooth_l1 = self._modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets)
        #  rpn_loss_box = tf.multiply(tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3])), 1)
        rpn_loss_box = tf.multiply(tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1,
                                                                reduction_indices=[1])),
                                   1.0)

        # R-CNN
        # classification loss
        cls_score = self.net.get_output('cls_score')
        label = tf.reshape(self.net.get_output('roi_data_3d')[2],[-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

        # bounding box regression L1 loss
        bbox_pred = self.net.get_output('bbox_pred')
        bbox_targets = self.net.get_output('roi_data_3d')[3]

        smooth_l1 = self._modified_smooth_l1(3.0, bbox_pred, bbox_targets)
        loss_box = tf.multiply(tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices=[1])),
                               1.0)

        # final loss
        loss = cross_entropy + loss_box + rpn_cross_entropy +  rpn_loss_box

        # optimizer and learning rate
        # global_step = tf.Variable(0, trainable=False)
        # lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
        #                                 cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
        #  momentum = cfg.TRAIN.MOMENTUM
        #  train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step=global_step)
        lr = 0.00001
        # train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)


        # iintialize variables
        sess.run(tf.global_variables_initializer())
        if not DEBUG:
            if self.pretrained_model is not None:
               print ('Loading pretrained model '
                      'weights from {:s}').format(self.pretrained_model)
               self.net.load(self.pretrained_model, sess, self.saver, True)

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            # get one batch
            blobs = data_layer.forward()

            # Make one SGD update
            feed_dict={self.net.image_data: blobs['image_data'],
                       self.net.lidar_bv_data: blobs['lidar_bv_data'],
                       self.net.im_info: blobs['im_info'],
                       self.net.keep_prob: 0.5,
                       self.net.gt_boxes: blobs['gt_boxes'],
                       self.net.gt_boxes_bv: blobs['gt_boxes_bv'],
                       self.net.gt_boxes_3d: blobs['gt_boxes_3d'],
                       self.net.gt_boxes_corners: blobs['gt_boxes_corners'],
                       self.net.calib: blobs['calib']}

            run_options = None
            run_metadata = None

            timer.tic()


            rpn_bbox_pred_out,rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, _ = sess.run([rpn_bbox_pred, rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, train_op],
                feed_dict=feed_dict,
                options=run_options,
                run_metadata=run_metadata)

            timer.toc()


            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(long(time.time() * 1000)) + '-train-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if DEBUG:
                cfg.TRAIN.DISPLAY = 1

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, lr: %f'%\
                        (iter+1, max_iters, rpn_loss_cls_value +  rpn_loss_box_value + loss_cls_value + loss_box_value ,rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, lr)
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                if vis:
                    cls_prob, bbox_pred_cnr, rpn_data, rpn_rois, rcnn_roi = sess.run([
                                             self.net.get_output('cls_prob'),
                                             self.net.get_output('bbox_pred'),
                                             self.net.get_output('rpn_data'),
                                             self.net.get_output('rpn_rois'),
                                             self.net.get_output('roi_data_3d')],
                                             feed_dict=feed_dict)
                    # self.net.get_output('rpn_bbox_pred'),
                    vis_detections(blobs['lidar_bv_data'], blobs['image_data'], blobs['calib'], bbox_pred_cnr, rpn_data, rpn_rois, rcnn_roi, cls_prob,  blobs['gt_boxes_3d'])

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

def vis_detections(lidar_bv, image, calib, bbox_pred_cnr, rpn_data, rpn_rois, rcnn_roi, scores, gt_boxes_3d):
    import matplotlib.pyplot as plt
    from utils.transform import lidar_3d_to_corners, corners_to_bv
    from fast_rcnn.bbox_transform import bbox_transform_inv_cnr
    from utils.draw import show_lidar_corners, show_image_boxes, scale_to_255
    from utils.cython_nms import nms, nms_new


    image = image.reshape((image.shape[1], image.shape[2], image.shape[3]))
    image += cfg.PIXEL_MEANS
    image = image.astype(np.uint8, copy=False)
    lidar_bv = lidar_bv.reshape((lidar_bv.shape[1], lidar_bv.shape[2], lidar_bv.shape[3]))[:,:,8]
    # visualize anchor_target_layer output
    rpn_anchors_3d = rpn_data[3][:,1:7]
    rpn_bv = rpn_data[2][:,1:5]
    # rpn_label = rpn_data[0]
    # print rpn_label.shape
    # print rpn_label[rpn_label==1]
    rpn_boxes_cnr = lidar_3d_to_corners(rpn_anchors_3d)
    img = show_lidar_corners(image, rpn_boxes_cnr, calib)
    img_bv = show_image_boxes(scale_to_255(lidar_bv, min=0, max=2), rpn_bv)

    print img.shape
    # plt.ion()
    plt.title('anchor target layer before regression')
    plt.subplot(211)
    plt.imshow(img_bv)
    plt.subplot(212)
    plt.imshow(img)
    plt.show()

    # visualize proposal_layer output
    boxes_3d = rpn_rois[2][:, 1:7]
    boxes_bv = rpn_rois[0][:, 0:5]
    boxes_img = rpn_rois[1][:, 0:5]


    # keep = nms(boxes_img, cfg.TEST.NMS)
    # boxes_img = boxes_img[keep]
    # boxes_3d = boxes_3d[keep]
    # boxes_cnr = lidar_3d_to_corners(boxes_3d[:100])
    print boxes_3d.shape
    print boxes_bv.shape
    # image_cnr = show_lidar_corners(image, boxes_cnr, calib)

    image_bv = show_image_boxes(lidar_bv, boxes_bv[:, 1:5])
    image_img = show_image_boxes(image, boxes_img[:, 1:5])
    plt.title('proposal_layer ')
    plt.subplot(211)
    plt.imshow(image_bv)
    plt.subplot(212)
    plt.imshow(image_img)
    plt.show()

    # visualize proposal_target_layer output
    # rois_image = rcnn_roi[1]
    # image2 = show_image_boxes(image, rois_image[:,1:5])
    # plt.title('proposal_target_layer ')
    # plt.imshow(image2)
    # plt.show()

    # # # visualize final
    # # #Apply bounding-box regression deltas
    # box_deltas = bbox_pred_cnr#[:boxes_3d.shape[0]]
    # boxes_3d = boxes_3d[:box_deltas.shape[0]]
    # boxes_cnr = lidar_3d_to_corners(boxes_3d)
    # # print '================'
    # # print box_deltas.shape
    # # print boxes_cnr.shape
    # boxes_cnr = bbox_transform_inv_cnr(boxes_cnr, box_deltas)
    # # print boxes_cnr.shape
    # boxes_bv = corners_to_bv(boxes_cnr)
    # # print scores

    # thresh = 0.15
    # for j in xrange(1, 2):
    #     inds = np.where(scores[:, j] > thresh)[0]
    #     cls_scores = scores[inds, j]
    #     cls_boxes = boxes_bv[inds, j*4:(j+1)*4]
    #     cls_boxes_cnr = boxes_cnr[inds, j*24:(j+1)*24]
    #     cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
    #         .astype(np.float32, copy=False)
    #     cls_dets_cnr = np.hstack((cls_boxes_cnr, cls_scores[:, np.newaxis])) \
    #         .astype(np.float32, copy=False)
    #     keep = nms(cls_dets, cfg.TEST.NMS)
    #     cls_dets = cls_dets[keep, :]
    #     cls_dets_cnr = cls_dets_cnr[keep, :]
    #     # project to image
    #     if np.any(cls_dets_cnr):
    #         img = show_lidar_corners(image, cls_dets_cnr[:,:24], calib)
    #         plt.title(str(j))
    #         plt.imshow(img)
    #         plt.show()

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            gdl_roidb.prepare_roidb(imdb)
        else:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            layer = GtDataLayer(roidb)
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)

    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=10000):
    """Train a Fast R-CNN network."""
    roidb = filter_roidb(roidb)
    saver = tf.train.Saver(max_to_keep=100)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sw = SolverWrapper(sess, saver, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)
        print 'Solving...'
        sw.train_model(sess, max_iters)
        print 'done solving'
