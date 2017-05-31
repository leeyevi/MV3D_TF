from __future__ import division
import _init_paths
import pykitti
import numpy as np
import cv2
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from utils.draw import show_lidar_corners, show_cam_corners, drawBox3D
from demo_mv import make_calib
from utils.draw import draw_lidar, draw_gt_boxes3d
import sys
import os
sys.path.append('/home/radmin/code/didi-udacity-2017/baseline-01')
from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED
import math

def read_objects(tracklet_file, num_frames):

    objects = []  #grouped by frames
    for n in range(num_frames): objects.append([])

    # read tracklets from file
    tracklets = parseXML(tracklet_file)
    num = len(tracklets)

    for n in range(num):
        tracklet = tracklets[n]

        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h,w,l = tracklet.size
        trackletBox = np.array([ # in velodyne coordinates around zero point and without orientation yet\
            [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2,-l/2], \
            [ w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2], \
            [ 0.0,  0.0,  0.0, 0.0,    h,     h,   h,   h]])

        # loop over all data in tracklet
        t  = tracklet.firstFrame
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:

            # determine if object is in the image; otherwise continue
            if truncation not in (TRUNC_IN_IMAGE, TRUNC_TRUNCATED):
               continue

            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]   # other rotations are 0 in all xml files I checked
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([\
              [np.cos(yaw), -np.sin(yaw), 0.0], \
              [np.sin(yaw),  np.cos(yaw), 0.0], \
              [        0.0,          0.0, 1.0]])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8,1)).T

            # calc yaw as seen from the camera (i.e. 0 degree = facing away from cam), as opposed to
            #   car-centered yaw (i.e. 0 degree = same orientation as car).
            #   makes quite a difference for objects in periphery!
            # Result is in [0, 2pi]
            x, y, z = translation
            yawVisual = ( yaw - np.arctan2(y, x) ) % (2*math.pi)

            o = type('', (), {})()
            o.box = cornerPosInVelo.transpose()
            o.type = tracklet.objectType
            o.lwh = [l, w, h]
            o.tracklet_id = n
            objects[t].append(o)
            t = t+1

    return objects

## objs to gt boxes ##
def obj_to_gt_boxes3d(objs):

    num        = len(objs)
    gt_boxes3d = np.zeros((num,8,3),dtype=np.float32)
    gt_labels  = np.zeros((num),    dtype=np.int32)
    gt_lwh  = np.zeros((num, 3),    dtype=np.float32)

    for n in range(num):
        obj = objs[n]
        b   = obj.box
        lwh = obj.lwh
        label = 1 #<todo>

        gt_labels [n]=label
        gt_boxes3d[n]=b
        gt_lwh[n] = lwh

    return  gt_boxes3d, gt_labels, gt_lwh


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    basedir = '/sdb-4T/raw_kitti/'
    date  = '2011_09_26'
    drive = '0064'
    calib = make_calib('/sdb-4T/raw_kitti/2011_09_26/object/0064/testing/calib/000000.txt')
    #  range = (150, 151, 1)
    # The range argument is optional - default is None, which loads the whole dataset
    dataset = pykitti.raw(basedir, date, drive) #, range(0, 50, 5))

    # Load some data
    dataset.load_calib()         # Calibration data are accessible as named tuples

    dataset.load_rgb()          # Left/right images are accessible as named tuples
    # dataset.load_velo()          # Each scan is a Nx4 array of [x,y,z,reflectance]

    tracklet_file = '/sdb-4T/raw_kitti/2011_09_26/2011_09_26_drive_0064_sync/tracklet_labels.xml'

    num_frames=len(dataset.rgb)  #154
    # num_frames = 154
    objects = read_objects(tracklet_file, num_frames)


    for n in range(num_frames):

        num = n
        gt_boxes3d, gt_labels, lwh = obj_to_gt_boxes3d(objects[num])
        corners = gt_boxes3d.transpose((0, 2, 1)).reshape((-1, 24))

        print lwh
        # lidar_cor = show_lidar_corners(dataset.rgb[num][0], corners, calib)
        # plt.imshow(lidar_cor)
        # plt.show()

        filename = os.path.join('/sdb-4T/raw_kitti/2011_09_26/object/0064/training/label_2', str(n).zfill(6) + '.npy')
        print filename
        label = np.hstack((np.ones(corners.shape[0]).reshape(-1, 1), lwh, corners))
        np.save(filename, label)


        # fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
        # draw_lidar(dataset.velo[num], fig=fig)
        # draw_gt_boxes3d(gt_boxes3d, fig=fig)

        # mlab.show()






