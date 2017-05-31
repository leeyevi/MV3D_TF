__author__ = 'Ryan Gao' # derived from honda.py by fyang

import datasets
import datasets.kitti_mv3d
import os
import time
import PIL
import datasets.imdb
import numpy as np
import scipy.sparse
from utils.cython_bbox import bbox_overlaps
from utils.boxes_grid import get_boxes_grid
import subprocess
import cPickle
from fast_rcnn.config import cfg
import math
from rpn_msr.generate_anchors import generate_anchors_bv
from utils.transform import camera_to_lidar_cnr, lidar_to_corners_single, computeCorners3D, lidar_3d_to_bv, lidar_cnr_to_3d

class kitti_raw(datasets.imdb):
    def __init__(self, image_set, kitti_path=None):
        datasets.imdb.__init__(self, image_set)
        self._image_set = image_set
        # self._kitti_path = '$Faster-RCNN_TF/data/KITTI'
        self._kitti_path = self._get_default_path() if kitti_path is None \
                            else kitti_path
        # self._data_path = '$Faster-RCNN_TF/data/KITTI/object'
        self._data_path = os.path.join(self._kitti_path, 'object')
        self._classes = ('__background__', 'Car')#, 'Pedestrian', 'Cyclist')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._lidar_ext = '.npy'
        self._subset = 'car'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler

        self._roidb_handler = self.gt_roidb

        self.config = {'top_k': 100000}

        # statistics for computing recall
        # self._num_boxes_all = np.zeros(self.num_classes, dtype=np.int)
        # self._num_boxes_covered = np.zeros(self.num_classes, dtype=np.int)
        # self._num_boxes_proposal = 0

        assert os.path.exists(self._kitti_path), \
                'KITTI path does not exist: {}'.format(self._kitti_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self.image_index[i])

    def lidar_path_at(self, i):
        """
        Return the absolute path to lidar i in the lidar sequence.
        """
        return self.lidar_path_from_index(self.image_index[i])

    def calib_at(self, i):
        """
        Return the calib sequence.
        """
        index = str(i).zfill(6)
        calib_ori =  self._load_kitti_calib(index)
        calib = np.zeros((4, 12))
        calib[0,:] = calib_ori['P2'].reshape(12)
        calib[1,:] = calib_ori['P3'].reshape(12)
        calib[2,:9] = calib_ori['R0'].reshape(9)
        calib[3,:] = calib_ori['Tr_velo2cam'].reshape(12)

        return calib

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # set the prefix
        if self._image_set == 'test':
            prefix = 'testing/image_2'
        else:
            prefix = 'training/image_2'
        # image_path = '$Faster-RCNN_TF/data/KITTI/object/training/image_2/000000.png'
        image_path = os.path.join(self._data_path, prefix, index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def lidar_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # set the prefix
        if self._image_set == 'test':
            prefix = 'testing/lidar_bv'
        else:
            prefix = 'training/lidar_bv'
        # lidar_bv_path = '$Faster-RCNN_TF/data/KITTI/object/training/lidar_bv/000000.npy'
        lidar_bv_path = os.path.join(self._data_path, prefix, index + self._lidar_ext)
        assert os.path.exists(lidar_bv_path), \
                'Path does not exist: {}'.format(lidar_bv_path)
        return lidar_bv_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # image_set_file = '$Faster-RCNN_TF/data/KITTI/ImageSets/train.txt'
        image_set_file = os.path.join(self._kitti_path, 'ImageSets',self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [x.rstrip('\n') for x in f.readlines()]

        print 'image sets length: ', len(image_index)
        return image_index

    def _get_default_path(self):
        """
        Return the default path where KITTI is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'KITTI')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_kitti_annotation(index)
                    for index in self.image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_kitti_calib(self, index):
        """
        load projection matrix

        """
        if self._image_set == 'test':
            prefix = 'testing/calib'
        else:
            prefix = 'training/calib'
        calib_dir = os.path.join(self._data_path, prefix, index + '.txt')
        

#         j = 0
        with open(calib_dir) as fi:
            lines = fi.readlines()
#             assert(len(lines) == 8)
        
#         obj = lines[0].strip().split(' ')[1:]
#         P0 = np.array(obj, dtype=np.float32)
#         obj = lines[1].strip().split(' ')[1:]
#         P1 = np.array(obj, dtype=np.float32)
        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)
#         obj = lines[6].strip().split(' ')[1:]
#         P0 = np.array(obj, dtype=np.float32)
            
        return {'P2' : P2.reshape(3,4),
                'P3' : P3.reshape(3,4),
                'R0' : R0.reshape(3,3),
                'Tr_velo2cam' : Tr_velo_to_cam.reshape(3, 4)}

    def _load_kitti_annotation(self, index):
        """
        Load image and bounding boxes info from txt file in the KITTI
        format.
        """
        # filename = '$Faster-RCNN_TF/data/KITTI/object/training/label_2/000000.txt'
        filename = os.path.join(self._data_path, 'training/label_2', index + '.npy')

        # calib
        calib = self._load_kitti_calib(index)

        lines = np.load(filename)

        num_objs = lines.shape[0]

        lwh = lines[:,1:4]
        boxes3D_corners = lines[:,4:]

        boxes_bv = corners_to_bv(boxes3D_corners)
        boxes = lidar_cnr_to_img(boxes3D_corners, calib[3], calib[2], calib[0])

        boxes3D_lidar = lidar_cnr_to_3d(boxes3D_corners, lwh)
        
        # TODO
        gt_classes = np.ones((num_objs), dtype=np.int32)
        overlaps = np.ones((num_objs, self.num_classes), dtype=np.float32)

        boxes.resize(num_objs+1, 4)
        boxes_bv.resize(num_objs, 4)
        boxes3D_lidar.resize(num_objs, 6)
        boxes3D_corners.resize(num_objs, 24)
        gt_classes.resize(num_objs)
        overlaps.resize(num_objs, self.num_classes)
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {
                'boxes' : boxes,
                'boxes_bv' : boxes_bv,
                'boxes_3D' : boxes3D_lidar,
                'boxes_corners' : boxes3D_corners,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _get_obj_level(self, obj):
        height = float(obj[7]) - float(obj[5]) + 1
        trucation = float(obj[1])
        occlusion = float(obj[2])
        if height >= 40 and trucation <= 0.15 and occlusion <= 0:
            return 1
        elif height >= 25 and trucation <= 0.3 and occlusion <= 1:
            return 2
        elif height >= 25 and trucation <= 0.5 and occlusion <= 2:
            return 3
        else:
            return 4

    def _write_kitti_results_file(self, all_boxes, all_boxes3D):
        # use_salt = self.config['use_salt']
        # comp_id = ''
        # if use_salt:
        #     comp_id += '{}'.format(os.getpid())

        path = os.path.join(datasets.ROOT_DIR, 'kitti/results', 'kitti_' + self._subset + '_' + self._image_set + '_' \
                                        + '-' + time.strftime('%m-%d-%H-%M-%S',time.localtime(time.time())), 'data')
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)
        for im_ind, index in enumerate(self.image_index):
            filename = os.path.join(path, index + '.txt')
            with open(filename, 'wt') as f:
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    # dets3D = all_boxes3D[cls_ind][im_ind]
                    # alphas = all_alphas[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the KITTI server expects 0-based indices
                    for k in xrange(dets.shape[0]):
                        # TODO
                        # alpha = dets3D[k, 0] - np.arctan2(dets3D[k, 4], dets3D[k, 6])
                        alpha = 0
                        f.write('{:s} -1 -1 {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} -1 -1 -1 -1 -1 -1 -1 -1\n' \
                                .format(cls.lower(), alpha, \
                                dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3]))
        return path

    def _write_corners_results_file(self, all_boxes, all_boxes3D):
        # use_salt = self.config['use_salt']
        # comp_id = ''
        # if use_salt:
        #     comp_id += '{}'.format(os.getpid())

        path = os.path.join(datasets.ROOT_DIR, 'kitti/results_cnr', 'kitti_' + self._subset + '_' + self._image_set + '_' \
                                        + '-' + time.strftime('%m-%d-%H-%M-%S',time.localtime(time.time())), 'data')
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)
        for im_ind, index in enumerate(self.image_index):
            filename = os.path.join(path, index + '.npy')
            with open(filename, 'wt') as f:
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    dets3D = all_boxes3D[cls_ind][im_ind]
                    # alphas = all_alphas[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the KITTI server expects 0-based indices
                    for k in xrange(dets.shape[0]):
                        obj = np.hstack((dets[k], dets3D[k, 1:]))
                        # print obj.shape
                        np.save(filename, obj)
                        # # TODO

        print 'Done'
        # return path

    def _do_eval(self, path, output_dir='output'):
        cmd = os.path.join(datasets.ROOT_DIR, 'kitti/eval/cpp/evaluate_object {}'.format(os.path.dirname(path)))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, all_boxes3D, output_dir):
        self._write_kitti_results_file(all_boxes, all_boxes3D)
        # path = self._write_kitti_results_file(all_boxes, all_boxes3D)
        # if self._image_set != 'test':
        #     self._do_eval(path)


# if __name__ == '__main__':
#     d = datasets.kitti_mv3d('train')
#     res = d.roidb
#     from IPython import embed; embed()
