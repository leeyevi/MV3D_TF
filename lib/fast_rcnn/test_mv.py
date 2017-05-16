from fast_rcnn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
from utils.cython_nms import nms, nms_new
from utils.boxes_grid import get_boxes_grid
from utils.transform import lidar_3d_to_corners, corners_to_bv, lidar_cnr_to_img_single, lidar_cnr_to_img
from utils.draw import show_lidar_corners, show_image_boxes, scale_to_255
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os
import math
from rpn_msr.generate import imdb_proposals_det
import tensorflow as tf
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv, bbox_transform_inv_cnr
import matplotlib.pyplot as plt
from tensorflow.python.client import timeline
import time

import mayavi.mlab as mlab

def draw_lidar(lidar, is_grid=False, is_top_region=False, fig=None):
    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    if fig is None: fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
    mlab.points3d(
        pxs, pys, pzs, prs,
        mode='point',  # 'point'  'sphere'
        colormap='gnuplot',  #'bone',  #'spectral',  #'copper',
        scale_factor=1,
        figure=fig)
    #draw grid
    if is_grid:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        for y in np.arange(-50,50,1):
            x1,y1,z1 = -50, y, 0
            x2,y2,z2 =  50, y, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

        for x in np.arange(-50,50,1):
            x1,y1,z1 = x,-50, 0
            x2,y2,z2 = x, 50, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

    #draw axis
    if 1:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        axes=np.array([
            [2.,0.,0.,0.],
            [0.,2.,0.,0.],
            [0.,0.,2.,0.],
        ],dtype=np.float64)
        fov=np.array([  ##<todo> : now is 45 deg. use actual setting later ...
            [20., 20., 0.,0.],
            [20.,-20., 0.,0.],
        ],dtype=np.float64)


        mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
        mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)

    #draw top_image feature area
    if is_top_region:
        x1 = TOP_X_MIN
        x2 = TOP_X_MAX
        y1 = TOP_Y_MIN
        y2 = TOP_Y_MAX
        mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

    mlab.orientation_axes()
    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991


def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=2):

    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]

        mlab.text3d(b[0,0], b[0,1], b[0,2], '%d'%n, scale=(1, 1, 1), color=color, figure=fig)
        for k in range(0,4):

            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991


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


    im_blob = im - cfg.PIXEL_MEANS
    lidar_bv_blob = bv

    im_blob = im_blob.reshape((1, im_blob.shape[0], im_blob.shape[1], im_blob.shape[2]))
    lidar_bv_blob = lidar_bv_blob.reshape((1, lidar_bv_blob.shape[0], lidar_bv_blob.shape[1], lidar_bv_blob.shape[2]))

    blobs = {'image_data': im_blob,
             'lidar_bv_data': lidar_bv_blob}

    im_scales = [1]

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


    conv5_3, deconv, rpn_cls_prob, rpn_cls_prob_reshape, rpn_cls_score_reshape, rpn_cls_score, cls_score, cls_prob, bbox_pred_cnr, rois = sess.run([
                                             net.get_output('conv5_3_2'),
                                             net.get_output('conv5_3'),
                                             net.get_output('rpn_cls_prob'),
                                             net.get_output('rpn_cls_prob_reshape'),
                                             net.get_output('rpn_cls_score_reshape'),
                                             net.get_output('rpn_cls_score'),
                                             net.get_output('cls_score'),
                                             net.get_output('cls_prob'),
                                             net.get_output('bbox_pred'),
                                             net.get_output('rois')],
                                             feed_dict=feed_dict)

    scores = cls_prob

    # plot featuremaps

    # print conv5_3.shape
    # # print deconv1.shape
    # activation = conv5_3
    # # featuremaps = activation.shape[3]
    # featuremaps = 48
    # plt.figure(1, figsize=(15,15))
    # for featuremap in range(featuremaps):
    #     plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
    #     # plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
    #     plt.axis('off')
    #     plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="jet")
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0.01)
    # plt.tight_layout(pad=0.1, h_pad=0.001, w_pad=0.001)
    # plt.show()

    # activation = deconv
    # print deconv.shape
    # # featuremaps = activation.shape[3]
    # featuremaps = 48
    # plt.figure(1, figsize=(15,15))
    # for featuremap in range(featuremaps):
    #     plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
    #     # plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
    #     plt.axis('off')
    #     plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="jet")
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0.01)
    # plt.tight_layout(pad=0.1, h_pad=0.001, w_pad=0.001)
    # plt.show()



    # print cls_score[np.where(cls_score[:,1] > 0)]
    # plt.hist(cls_score[:,1], bins=25)
    # plt.show()
    # plt.hist(scores[:,1], bins=25)
    # plt.show()

    assert len(im_scales) == 1, "Only single-image batch implemented"
    boxes_3d = rois[2][:, 1:7]

    # Apply bounding-box regression deltas
    box_deltas = bbox_pred_cnr
    boxes_cnr = lidar_3d_to_corners(boxes_3d)

    # img_boxes = lidar_cnr_to_img(boxes_cnr, calib[3], calib[2], calib[0])
    # img = show_image_boxes(im, img_boxes)
    # plt.imshow(img)
    # plt.show()


    # !! Important
    # ! Not apply corner regression
    pred_boxes_cnr = np.hstack((boxes_cnr, boxes_cnr))
    # apply corner regression
    pred_boxes_cnr_r = bbox_transform_inv_cnr(boxes_cnr, box_deltas)


    #  preject corners to lidar_bv
    pred_boxes_bv = corners_to_bv(pred_boxes_cnr)


    return scores, pred_boxes_bv, pred_boxes_cnr, pred_boxes_cnr_r

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


    # conv1_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv1_1")
    # conv1_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv1_2")
    # conv2_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv2_1")
    # conv2_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv2_2")
    # conv3_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv3_1")
    # conv3_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv3_2")
    # conv3_3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv3_3")
    # conv4_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv4_1")
    # conv4_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv4_2")
    # conv4_3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv4_3")
    # conv5_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv5_1")
    # conv5_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv5_2")
    # conv5_3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv5_3")

    # rpn_w = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rpn_conv/3x3")[0]
    # rpn_b = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rpn_conv/3x3")[1]
    # rpn_w2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rpn_cls_score")[0]
    # rpn_b2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rpn_cls_score")[1]
    # rpn_w3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rpn_bbox_pred")[0]
    # rpn_b3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rpn_bbox_pred")[1]

    # weights = {
    # 'conv1_1' : {"weights" : conv1_1[0].eval(session=sess), "biases": conv1_1[1].eval(session=sess)},
    # 'conv1_2' : {"weights" : conv1_2[0].eval(session=sess), "biases": conv1_2[1].eval(session=sess)},
    # 'conv2_1' : {"weights" : conv2_1[0].eval(session=sess), "biases": conv2_1[1].eval(session=sess)},
    # 'conv2_2' : {"weights" : conv2_2[0].eval(session=sess), "biases": conv2_2[1].eval(session=sess)},
    # 'conv3_1' : {"weights" : conv3_1[0].eval(session=sess), "biases": conv3_1[1].eval(session=sess)},
    # 'conv3_2' : {"weights" : conv3_2[0].eval(session=sess), "biases": conv3_2[1].eval(session=sess)},
    # 'conv3_3' : {"weights" : conv3_3[0].eval(session=sess), "biases": conv3_3[1].eval(session=sess)},
    # 'conv4_1' : {"weights" : conv4_1[0].eval(session=sess), "biases": conv4_1[1].eval(session=sess)},
    # 'conv4_2' : {"weights" : conv4_2[0].eval(session=sess), "biases": conv4_2[1].eval(session=sess)},
    # 'conv4_3' : {"weights" : conv4_3[0].eval(session=sess), "biases": conv4_3[1].eval(session=sess)},
    # 'conv5_1' : {"weights" : conv5_1[0].eval(session=sess), "biases": conv5_1[1].eval(session=sess)},
    # 'conv5_2' : {"weights" : conv5_2[0].eval(session=sess), "biases": conv5_2[1].eval(session=sess)},
    # 'conv5_3' : {"weights" : conv5_3[0].eval(session=sess), "biases": conv5_3[1].eval(session=sess)},

    # 'rpn_conv/3x3' : {"weights" : rpn_w.eval(session=sess), "biases": rpn_b.eval(session=sess)},
    # 'rpn_cls_score' : {"weights" : rpn_w2.eval(session=sess), "biases": rpn_b2.eval(session=sess)},
    # 'rpn_bbox_pred' : {"weights" : rpn_w3.eval(session=sess), "biases": rpn_b3.eval(session=sess)},
    # }
    # # print rpn_w.eval(session=sess)
    # np.save('rpn_data.npy', weights)


    # deconv2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="deconv_4x_1")[0]
    # shape_conv5_3 = conv5_3.get_shape().as_list()
    # shape1 = deconv1.get_shape().as_list()
    # shape2 = deconv2.get_shape().as_list()
    # print 'conv5_3 shape', shape_conv5_3
    # print 'deconv_2x_1 shape', shape1
    # print 'deconv_4x_1 shape', shape2


    for i in xrange(num_images):

        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None

        im = cv2.imread(imdb.image_path_at(i))
        bv = np.load(imdb.lidar_path_at(i))
        calib = imdb.calib_at(i)



        print "Inference: ", imdb.lidar_path_at(i)


        _t['im_detect'].tic()
        scores, boxes_bv, boxes_cnr, boxes_cnr_r = box_detect(sess, net, im, bv, calib,  box_proposals)
        _t['im_detect'].toc()


        _t['misc'].tic()
        if vis:
            image = im[:, :, (2, 1, 0)]
            plt.cla()
            plt.imshow(image)

        thresh = 0.05

        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes_bv[inds, j*4:(j+1)*4]
            cls_boxes_cnr = boxes_cnr[inds, j*24:(j+1)*24]

            cls_boxes_cnr_r = boxes_cnr_r[inds, j*24:(j+1)*24]

            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            cls_dets_cnr = np.hstack((cls_boxes_cnr, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            cls_dets_cnr_r = np.hstack((cls_boxes_cnr_r, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            # print "scores: ", cls_scores
            # print "cls_dets : ", cls_dets.shape

            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            cls_dets_cnr = cls_dets_cnr[keep, :]
            cls_dets_cnr_r = cls_dets_cnr_r[keep, :]
            cls_scores = cls_scores[keep]

            # project to image
            if np.any(cls_dets_cnr):

                plt.rcParams['figure.figsize'] = (10, 10)

                img_boxes = lidar_cnr_to_img(cls_dets_cnr[:,:24], calib[3], calib[2], calib[0])
                img = show_image_boxes(im, img_boxes)
                plt.imshow(img)
                #plt.show()

                print cls_dets_cnr.shape
                image_bv = show_image_boxes(scale_to_255(bv[:,:,8], min=0, max=2), cls_dets[:, :4])
                image_cnr = show_lidar_corners(im, cls_dets_cnr[:,:24], calib)


                if 1:
                    import mayavi.mlab as mlab

                    filename = os.path.join(imdb.lidar_path_at(i)[:-19], 'velodyne', str(i).zfill(6)+'.bin')

                    print filename
                    scan = np.fromfile(filename, dtype=np.float32)
                    scan = scan.reshape((-1, 4))
                    corners = cls_dets_cnr[:,:24].reshape((-1, 3, 8)).transpose((0, 2, 1))

                    corners_r = cls_dets_cnr_r[:,:24].reshape((-1, 3, 8)).transpose((0, 2, 1))
                    print corners_r
                    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
                    draw_lidar(scan, fig=fig)
                    draw_gt_boxes3d(corners, fig=fig)
                    draw_gt_boxes3d(corners_r, color = (1,0,1),fig=fig)
                    mlab.show()

                plt.subplot(211)
                plt.title('bv proposal')
                plt.imshow(image_bv, cmap='jet')
                plt.subplot(212)
                plt.imshow(image_cnr)
                plt.show()

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
    with open(det_cnr_file, 'wb') as f:
        cPickle.dump(all_boxes_cnr, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, all_boxes_cnr, output_dir)

