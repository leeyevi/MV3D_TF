import _init_paths
from fast_rcnn.config import cfg
import argparse
from utils.timer import Timer
import numpy as np
import cv2
from utils.cython_nms import nms
from utils.transform import lidar_3d_to_corners, corners_to_bv, lidar_cnr_to_img_single, lidar_cnr_to_img
from utils.draw import show_lidar_corners, show_image_boxes, scale_to_255
import cPickle
from utils.blob import im_list_to_blob
import os
import math
from networks.factory import get_network
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import mayavi.mlab as mlab
from utils.draw import draw_lidar, draw_gt_boxes3d
from fast_rcnn.test_mv import box_detect
from read_lidar import point_cloud_2_top

plt.rcParams['figure.figsize'] = (10, 10)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

CLASSES = ('__background__',
           'car')

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


def make_calib(calib_dir):


    with open(calib_dir) as fi:
        lines = fi.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    calib = np.empty((4, 12))
    calib[0,:] = P2.reshape(12)
    calib[1,:] = P3.reshape(12)
    calib[2,:9] = R0.reshape(9)
    calib[3,:] = Tr_velo_to_cam.reshape(12)

    return calib


def make_bird_view(velo_file):

    print("Processing: ", velo_file)
    scan = np.fromfile(velo_file, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    # bird_view = point_cloud_2_top(scan, res=0.1, zres=0.3,
    #                                side_range=(-30., 30.),  # left-most to right-most
    #                                fwd_range=(0, 60.),  # back-most to forward-most
    #                                height_range=(-2., 0.4))
    bird_view = []
    return scan, bird_view

def demo(sess, net, root_dir, image_name):
    """Test a Fast R-CNN network on an image database."""

    # Load the demo image

    im_file = os.path.join(root_dir, 'image_2' , image_name+'.png')
    velo_file = os.path.join(root_dir, 'velodyne', image_name+'.bin')
    calib_file = os.path.join(root_dir, 'calib', '000000'+'.txt')
    bv_file = os.path.join(root_dir, 'lidar_bv', image_name+'.npy')

    im = cv2.imread(im_file)
    velo = make_bird_view(velo_file)[0]
    bv = np.load(bv_file)
    calib = make_calib(calib_file)

    plt.imshow(bv[:,:,5])
    plt.show

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes_bv, boxes_cnr, boxes_cnr_r = box_detect(sess, net, im, bv, calib)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes_bv.shape[0])

    # Visualize detections for each class
    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    # plt.imshow(im)
    # plt.show()

    CONF_THRESH = 0.1
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background

        inds = np.where(scores[:, cls_ind] > CONF_THRESH)[0]
        cls_boxes = boxes_bv[inds, 4*cls_ind:4*(cls_ind + 1)]
        cls_boxes_cnr = boxes_cnr[inds, cls_ind*24:(cls_ind+1)*24]
        cls_boxes_cnr_r = boxes_cnr_r[inds, cls_ind*24:(cls_ind+1)*24]
        cls_scores = scores[inds, cls_ind]

        # cls_boxes = boxes_bv[:, 4*cls_ind:4*(cls_ind + 1)]
        # cls_boxes_cnr = boxes_cnr[:, cls_ind*24:(cls_ind+1)*24]
        # cls_boxes_cnr_r = boxes_cnr_r[:, cls_ind*24:(cls_ind+1)*24]
        # cls_scores = scores[:, cls_ind]

        cls_dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
        cls_dets_cnr = np.hstack((cls_boxes_cnr, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
        cls_dets_cnr_r = np.hstack((cls_boxes_cnr_r, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)

        keep = nms(cls_dets, NMS_THRESH)
        cls_dets = cls_dets[keep, :]
        cls_dets_cnr = cls_dets_cnr[keep, :]
        cls_dets_cnr_r = cls_dets_cnr_r[keep, :]
        cls_scores = cls_scores[keep]
        print cls_dets[:,:4]
        print cls_scores

        # vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)

        # project to image
        # if np.any(cls_dets_cnr):

        # img_boxes = lidar_cnr_to_img(cls_dets_cnr_r[:,:24], calib[3], calib[2], calib[0])
        # # print img_boxes
        # img = show_image_boxes(im, img_boxes)
        # plt.imshow(img)
        # cv2.imwrite('examples/' + image_name+'.png', img)

        print cls_dets_cnr_r.shape
        image_bv = show_image_boxes(scale_to_255(bv[:,:,8], min=0, max=2), cls_dets[:, :4])
        image_cnr = show_lidar_corners(im, cls_dets_cnr_r[:,:24], calib)

        cv2.imwrite(image_name+'.png', image_cnr)
        # plt.imshow()

        if 1:

            corners = cls_dets_cnr[:,:24].reshape((-1, 3, 8)).transpose((0, 2, 1))
            corners_r = cls_dets_cnr_r[:,:24].reshape((-1, 3, 8)).transpose((0, 2, 1))
            fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
            draw_lidar(velo, fig=fig)
            # draw_gt_boxes3d(corners, fig=fig)
            draw_gt_boxes3d(corners_r, color = (1,0,1),fig=fig)
            mlab.savefig('lidar' + image_name+'.png', figure=fig)
            mlab.close()
            # mlab.show()

            # plt.subplot(211)
            # plt.title('bv proposal')
            # plt.imshow(image_bv, cmap='jet')
            # plt.subplot(212)
            # plt.imshow(image_cnr)
            # plt.show()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='MV3D_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))

    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    print net
    # load model
    saver = tf.train.Saver(max_to_keep=5)
    net.load(args.model, sess, saver, True)
    print '\n\nLoaded network {:s}'.format(args.model)

    # im_names = ['000456', '000542', '001150',
    #             '001763', '004545']

    root_dir = '/sdb-4T/raw_kitti/2011_09_26/object/0064/training'
    num = len(os.listdir(os.path.join(root_dir, 'image_2')))

    for im_name in range(449, 570):
        # if im_name == 124:
        #     continue
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(sess, net, root_dir, str(im_name).zfill(6))

    plt.show()
