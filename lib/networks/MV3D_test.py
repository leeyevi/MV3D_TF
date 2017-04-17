import tensorflow as tf
from networks.network import Network

n_classes = 2
_feat_stride = [4,]
anchor_scales = [1, 1] 

class MV3D_test(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.lidar_bv_data = tf.placeholder(tf.float32, shape=[None, None, None, 24])
        self.image_data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'lidar_bv_data':self.lidar_bv_data,
                            'image_data':self.image_data,
                            'im_info':self.im_info})
        self.trainable = trainable
        self.setup()

    def setup(self):
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

        (self.feed('conv5_3')
             .deconv(shape=None, c_o=512, stride=2, ksize=3,  name='deconv_2x_1')
             .conv(3,3,512,1,1,name='rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*2*2,1,1,padding='VALID',relu = False,name='rpn_cls_score'))

        (self.feed('rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*2*6,1,1,padding='VALID',relu = False,name='rpn_bbox_pred'))

        (self.feed('rpn_cls_score')
             .reshape_layer(2,name = 'rpn_cls_score_reshape')
             .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
             .reshape_layer(len(anchor_scales)*2*2,name = 'rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
             .proposal_layer_3d(_feat_stride, 'TEST', name = 'rois'))

        (self.feed('conv5_3')
             .deconv(shape=None, c_o=512, stride=4, ksize=3, name='deconv_4x_1'))

        (self.feed('deconv_4x_1', 'rois')
             .roi_pool(7, 7, 1.0/4, name='pool_5')
             .fc(4096, name='fc6')
             .fc(4096, name='fc7')
             .fc(n_classes, relu=False, name='cls_score')
             .softmax(name='cls_prob'))

        (self.feed('fc7')
             .fc(n_classes*24, relu=False, name='bbox_pred'))

