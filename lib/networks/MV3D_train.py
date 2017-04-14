import tensorflow as tf
from networks.network import Network

#  n_classes = 21
#  _feat_stride = [16,]
#  anchor_scales = [8, 16, 32]
n_classes = 4 # car, pedes, cyclist, dontcare
_feat_stride = [4]
anchor_scales = [1, 1] 

class MV3D_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.lidar_bv = tf.placeholder(tf.float32, shape=[None, None, None, 24])
        self.image = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.gt_boxes_bv = tf.placeholder(tf.float32, shape=[None, 5])
        self.gt_boxes_3d = tf.placeholder(tf.float32, shape=[None, 7])
        self.ryaw = tf.placeholder(tf.float32, shape=[None, 2])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'lidar_bv':self.lidar_bv, 'rgb':self.image,
                            'im_info':self.im_info,
                            'gt_boxes':self.gt_boxes, 
                            'gt_boxes_bv':self.gt_boxes_bv,  'gt_boxes_3d': self.gt_boxes_3d})
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
        (self.feed('lidar_bv')
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
        #  (self.feed('rgb')
             #  .conv(3, 3, 64, 1, 1, name='conv1_1_2', trainable=False)
             #  .conv(3, 3, 64, 1, 1, name='conv1_2_2', trainable=False)
             #  .max_pool(2, 2, 2, 2, padding='VALID', name='pool1_2')
             #  .conv(3, 3, 128, 1, 1, name='conv2_1_2', trainable=False)
             #  .conv(3, 3, 128, 1, 1, name='conv2_2_2', trainable=False)
             #  .max_pool(2, 2, 2, 2, padding='VALID', name='pool2_2')
             #  .conv(3, 3, 256, 1, 1, name='conv3_1_2')
             #  .conv(3, 3, 256, 1, 1, name='conv3_2_2')
             #  .conv(3, 3, 256, 1, 1, name='conv3_3_2')
             #  .max_pool(2, 2, 2, 2, padding='VALID', name='pool3_2')
             #  .conv(3, 3, 512, 1, 1, name='conv4_1_2')
             #  .conv(3, 3, 512, 1, 1, name='conv4_2_2')
             #  .conv(3, 3, 512, 1, 1, name='conv4_3_2')
             #  .conv(3, 3, 512, 1, 1, name='conv5_1_2')
             #  .conv(3, 3, 512, 1, 1, name='conv5_2_2')
             #  .conv(3, 3, 512, 1, 1, name='conv5_3_2'))

        #========= RPN ============
        (self.feed('conv5_3')
             .deconv(shape=None, c_o=256, stride=2, ksize=3,  name='deconv_2x_1')
             .conv(3,3,512,1,1,name='rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*2*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score'))

        (self.feed('rpn_cls_score','gt_boxes_bv','im_info','lidar_bv')
             .anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data' )) # 4 downsample

        # Loss of rpn_cls & rpn_boxes
        # ancho_num * xyzhlw
        # offset
        (self.feed('rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*2*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred'))

        #========= RoI Proposal ============
        # Lidar Bird View
        (self.feed('rpn_cls_score')
             .reshape_layer(2,name = 'rpn_cls_score_reshape')
             .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
             .reshape_layer(len(anchor_scales)*2*2,name = 'rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
             .proposal_layer(_feat_stride, anchor_scales, 'TRAIN',name = 'rpn_rois'))

        (self.feed('rpn_rois', 'gt_boxes_bv','gt_boxes_3d')
             .proposal_target_layer_3d(n_classes, name='roi_data_3d'))

        (self.feed('roi_data_3d')
             .proposal_transform(target='bv', name='roi_data_1'))

        (self.feed('conv5_3')
              .deconv(shape=None, c_o=512, stride=2, ksize=3, name='deconv_4x_1'))

        #========= RoI Proposal ============
        # RGB Mono
        #  (self.feed('roi_data_3d')
             #  .proposal_target_transform(target='image', name='roi_data_2'))

        #  (self.feed('conv5_3_2')
             #  .deconv(c_o=256, stride=2, ksize=3, name='deconv_2x_2'))
        #       #  .roi_pool(7, 7, 1.0/16, name='pool_5_1')

        #========= RCNN ============
        # (self.feed('conv5_3', 'roi_data_1')
        (self.feed('deconv_4x_1', 'roi_data_1')
             .roi_pool(7, 7, 1.0/8, name='pool_5')
             .fc(4096, name='fc6')
             .dropout(0.5, name='drop6')
             .fc(4096, name='fc7')
             .dropout(0.5, name='drop7')
             .fc(n_classes, relu=False, name='cls_score')
             .softmax(name='cls_prob'))

        (self.feed('drop7')
             .fc(n_classes*6, relu=False, name='bbox_pred'))

        # lidar
        #  (self.feed('deconv_4x_1', 'roi_data_1')
             #  .roi_pool(7, 7, 1.0/16, name='pool_5_1')
             #  .fc(4096, name='fc6_1')
             #  .dropout(0.5, name='drop6_1')
             #  .fc(4096, name='fc7_2')
             #  .dropout(0.5, name='drop7_1'))

        # rgb mono
        #  (self.feed('deconv_2x_2', 'roi_data_2')
             #  .roi_pool(7, 7, 1.0/16, name='pool_5_2')
             #  .fc(2048, name='fc6_1')
             #  .dropout(0.5, name='drop6_1')
             #  .fc(2048, name='fc7_2')
             #  .dropout(0.5, name='drop7_2'))

        # Late Fusion
        #  (self.feed('drop7_1', 'drop7_2')
             #  .concat(axis=0, name='fusion')
             #  .fc(n_classes, relu=False, name='cls_score')
             #  .softmax(name='cls_prob'))

        #  (self.feed('drop7')
             #  .fc(n_classes*4, relu=False, name='bbox_pred'))

