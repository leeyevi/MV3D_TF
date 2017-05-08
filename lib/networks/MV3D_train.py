import tensorflow as tf
from networks.network import Network

#  n_classes = 21
#  _feat_stride = [16,]
#  anchor_scales = [8, 16, 32]
n_classes = 2 # car, pedes, cyclist, dontcare
_feat_stride = [8, 8]
#  anchor_scales = [0.5, 1, 2]
anchor_scales = [1.0, 1.0]

class MV3D_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.lidar_bv_data = tf.placeholder(tf.float32, shape=[None, None, None, 9])
        self.image_data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.gt_boxes_bv = tf.placeholder(tf.float32, shape=[None, 5])
        self.gt_boxes_3d = tf.placeholder(tf.float32, shape=[None, 7])
        self.gt_boxes_corners = tf.placeholder(tf.float32, shape=[None, 25])
        self.calib = tf.placeholder(tf.float32, shape=[None, 12])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'lidar_bv_data':self.lidar_bv_data,
                            'image_data':self.image_data,
                            'calib' : self.calib,
                            'im_info':self.im_info,
                            'gt_boxes':self.gt_boxes,
                            'gt_boxes_bv':self.gt_boxes_bv,
                            'gt_boxes_3d': self.gt_boxes_3d,
                            'gt_boxes_corners': self.gt_boxes_corners})
        self.trainable = trainable
        self.setup()

        # create ops and placeholders for bbox normalization process
        with tf.variable_scope('bbox_pred', reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

            self.bbox_weights_assign = weights.assign(self.bbox_weights)
            self.bbox_bias_assign = biases.assign(self.bbox_biases)

    def setup(self):
        # Lidar Bird View
        (self.feed('lidar_bv_data')
             .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
             .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
             .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3'))
        # RGB
        (self.feed('image_data')
              .conv(3, 3, 64, 1, 1, name='conv1_1_2', trainable=False)
              .conv(3, 3, 64, 1, 1, name='conv1_2_2', trainable=False)
              .max_pool(2, 2, 2, 2, padding='VALID', name='pool1_2')
              .conv(3, 3, 128, 1, 1, name='conv2_1_2', trainable=False)
              .conv(3, 3, 128, 1, 1, name='conv2_2_2', trainable=False)
              .max_pool(2, 2, 2, 2, padding='VALID', name='pool2_2')
              .conv(3, 3, 256, 1, 1, name='conv3_1_2')
              .conv(3, 3, 256, 1, 1, name='conv3_2_2')
              .conv(3, 3, 256, 1, 1, name='conv3_3_2')
              .max_pool(2, 2, 2, 2, padding='VALID', name='pool3_2')
              .conv(3, 3, 512, 1, 1, name='conv4_1_2')
              .conv(3, 3, 512, 1, 1, name='conv4_2_2')
              .conv(3, 3, 512, 1, 1, name='conv4_3_2')
              .conv(3, 3, 512, 1, 1, name='conv5_1_2')
              .conv(3, 3, 512, 1, 1, name='conv5_2_2')
              .conv(3, 3, 512, 1, 1, name='conv5_3_2'))

        #========= RPN ============
                     # 
        (self.feed('conv5_3')
             # .deconv(shape=None, c_o=512, stride=2, ksize=3,  name='deconv_2x_1')
             .conv(3,3,512,1,1,name='rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*2*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score'))

        (self.feed('rpn_cls_score','gt_boxes_bv', 'gt_boxes_3d', 'im_info')
             .anchor_target_layer(_feat_stride[0], anchor_scales, name = 'rpn_data' )) # 4 downsample

        # Loss of rpn_cls & rpn_boxes
        # ancho_num * xyzhlw
        # offset
        (self.feed('rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*2*6, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred'))

        #========= RoI Proposal ============
        # Lidar Bird View
        (self.feed('rpn_cls_score')
             .reshape_layer(2,name = 'rpn_cls_score_reshape')
             .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
             .reshape_layer(len(anchor_scales)*2*2,name = 'rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info', 'calib')
             .proposal_layer_3d(_feat_stride[0], 'TRAIN', name = 'rpn_rois'))

        (self.feed('rpn_rois', 'gt_boxes_bv', 'gt_boxes_3d', 'gt_boxes_corners', 'calib')
             .proposal_target_layer_3d(n_classes, name='roi_data_3d'))
            # return
            # 1. rois: lidar_bv (nx4)
            # #3. rois: image (nx4)
            # 4. labels (nx1)
            # 5. bbox_targets (nx24)

        (self.feed('roi_data_3d')
             .proposal_transform(target='img', name='roi_data_img'))
        (self.feed('roi_data_3d')
             .proposal_transform(target='bv', name='roi_data_bv'))

        (self.feed('conv5_3')
             .deconv(shape=None, c_o=512, stride=4, ksize=3, name='deconv_4x_1'))

        (self.feed('conv5_3_2')
             .deconv(shape=None, c_o=512, stride=2, ksize=3, name='deconv_2x_2'))

        #========= RoI Proposal ============

        #  (self.feed('conv5_3_2')
             #  .deconv(c_o=256, stride=2, ksize=3, name='deconv_2x_2'))
        #       #  .roi_pool(7, 7, 1.0/16, name='pool_5_1')

        #========= RCNN ============
        # (self.feed('conv5_3', 'roi_data_1')

        # lidar bv
        #  (self.feed('deconv_4x_1', 'roi_data_bv')
            #  .roi_pool(7, 7, 1.0/2, name='pool_5')
            #  .fc(4096, name='fc6')
            #  .dropout(0.5, name='drop6')
            #  .fc(4096, name='fc7')
            #  .dropout(0.5, name='drop7')
            #  .fc(n_classes, relu=False, name='cls_score')
            #  .softmax(name='cls_prob'))

        #  (self.feed('drop7')
            #  .fc(n_classes*24, relu=False, name='bbox_pred')) # (x0-x7,y0-y7,z0-z7)

        # lidar_bv
        (self.feed('deconv_4x_1', 'roi_data_bv')
        # (self.feed('conv5_3', 'roi_data_bv')
             .roi_pool(7, 7, 1.0/2, name='pool_5')
             .fc(2048, name='fc6_1')
             .dropout(self.keep_prob, name='drop6'))
             #  .fc(2048, name='fc7_1')
              # .dropout(self.keep_prob, name='drop7'))

        # image
        (self.feed('deconv_2x_2', 'roi_data_img')
        # (self.feed('conv5_3_2', 'roi_data_img')
             .roi_pool(7, 7, 1.0/4, name='pool_5')
             .fc(2048, name='fc6_2')
             .dropout(self.keep_prob, name='drop6_1'))
             #  .fc(2048, name='fc7_2')
             #  .dropout(self.keep_prob, name='drop7_2'))

        # fusion
        (self.feed('drop6', 'drop6_1')
             .concat(axis=1, name='concat1')
             .fc(4096, name='fc7')
             .dropout(self.keep_prob, name='drop7')
             .fc(n_classes, relu=False, name='cls_score')
             .softmax(name='cls_prob'))

        (self.feed('drop7')
             .fc(n_classes*24, relu=False, name='bbox_pred')) # (x0-x7,y0-y7,z0-z7)

        # fusion
        #  (self.feed('drop7', 'drop7_2')
             #  .concat(axis=1, name='concat1')
             #  .fc(n_classes, relu=False, name='cls_score')
             #  .softmax(name='cls_prob'))

        #  (self.feed('drop7', 'drop7_2')
             #  .concat(axis=1, name='concat2')
             #  .fc(n_classes*24, relu=False, name='bbox_pred')) # (x0-x7,y0-y7,z0-z7)

