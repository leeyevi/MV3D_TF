import numpy as np
import tensorflow as tf
import roi_pooling_layer.roi_pooling_op as roi_pool_op
import roi_pooling_layer.roi_pooling_op_grad
from rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from rpn_msr.proposal_layer_tf import proposal_layer_3d as proposal_layer_py_3d
from rpn_msr.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer_tf import proposal_target_layer as proposal_target_layer_py
from rpn_msr.proposal_target_layer_tf import proposal_target_layer_3d as proposal_target_layer_py_3d
from fast_rcnn.config import cfg

DEFAULT_PADDING = 'SAME'
# TOP_X_MAX = 70.3
# TOP_X_MIN = 0
# TOP_Y_MIN = -40
# TOP_Y_MAX = 40
# RES = 0.1
# LIDAR_HEIGHT = 1.73
# CAR_HEIGHT = 1.56


def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, saver, ignore_missing=False):
        if data_path.endswith('.ckpt.meta'):
            print '========================'
            saver = tf.train.import_meta_graph(data_path)
            saver.restore(session, data_path[:-5])

        else:
            data_dict = np.load(data_path).item()
            for key in data_dict:
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print "assign pretrain model "+subkey+ " to "+key
                        except ValueError:
                            print "ignore "+key
                            if not ignore_missing:

                                raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')


    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight_decay')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
        return regularizer

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable)
            biases = self.make_var('biases', [c_o], init_biases, trainable)

            if group==1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)
            return tf.nn.bias_add(conv, biases, name=scope.name)

    @layer
    def deconv(self, input, shape, c_o, ksize=4, stride = 2, name = 'upconv', biased=False, relu=True, padding=DEFAULT_PADDING,
             trainable=True):
        """ up-conv"""
        self.validate_padding(padding)

        c_in = input.get_shape()[3].value
        in_shape = tf.shape(input)
        if shape is None:
            # h = ((in_shape[1] - 1) * stride) + 1
            # w = ((in_shape[2] - 1) * stride) + 1
            h = ((in_shape[1] ) * stride)
            w = ((in_shape[2] ) * stride)
            new_shape = [in_shape[0], h, w, c_o]
        else:
            new_shape = [in_shape[0], shape[1], shape[2], c_o]
        output_shape = tf.stack(new_shape)

        filter_shape = [ksize, ksize, c_o, c_in]

        with tf.variable_scope(name) as scope:
            # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
            filters = self.make_var('weights', filter_shape, init_weights, trainable)
                                   # regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            deconv = tf.nn.conv2d_transpose(input, filters, output_shape,
                                            strides=[1, stride, stride, 1], padding=DEFAULT_PADDING, name=scope.name)
            # coz de-conv losses shape info, use reshape to re-gain shape
            deconv = tf.reshape(deconv, new_shape)

            if biased:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                if relu:
                    bias = tf.nn.bias_add(deconv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(deconv, biases)
            else:
                if relu:
                    return tf.nn.relu(deconv)
                return deconv


    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def roi_pool(self, input, pooled_height, pooled_width, spatial_scale, name):
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]

        print input
        return roi_pool_op.roi_pool(input[0], input[1],
                                    pooled_height,
                                    pooled_width,
                                    spatial_scale,
                                    name=name)[0]

    @layer
    def proposal_layer(self, input, _feat_stride, anchor_scales, cfg_key, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        return tf.reshape(tf.py_func(proposal_layer_py,[input[0],input[1],input[2], cfg_key, _feat_stride, anchor_scales], [tf.float32]),[-1,5],name =name)

    @layer
    def proposal_layer_3d(self, input, _feat_stride, cfg_key, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:
            rpn_rois_bv, rpn_rois_img, rpn_rois_3d = tf.py_func(proposal_layer_py_3d,[input[0],input[1],input[2], input[3], cfg_key, _feat_stride], [tf.float32, tf.float32, tf.float32])
            rpn_rois_bv = tf.reshape(rpn_rois_bv,[-1,5] , name = 'rpn_rois_bv')
            rpn_rois_img = tf.reshape(rpn_rois_img,[-1,5] , name = 'rpn_rois_img')
            rpn_rois_3d = tf.reshape(rpn_rois_3d,[-1,7] , name = 'rpn_rois_3d')

        #if cfg_key == 'TRAIN':
        #    return rpn_rois_bv, rpn_rois_3d
        #else :
        return rpn_rois_bv, rpn_rois_img, rpn_rois_3d, rpn_rois_3d


    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        # gt_boxes_bv = lidar_to_top(input[1])
        with tf.variable_scope(name) as scope:

            rpn_labels,rpn_bbox_targets, rpn_rois_bv, rpn_rois_3d = \
            tf.py_func(anchor_target_layer_py,[input[0],input[1],input[2],input[3], _feat_stride, anchor_scales],[tf.float32,tf.float32, tf.float32, tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels')
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets')
            rpn_rois_bv = tf.reshape(rpn_rois_bv,[-1,5] , name = 'rpn_rois_bv')
            # rpn_rois_img = tf.reshape(rpn_rois_img,[-1,5] , name = 'rpn_rois_img')
            rpn_rois_3d = tf.reshape(rpn_rois_3d,[-1,7] , name = 'rpn_rois_3d')
            return rpn_labels, rpn_bbox_targets, rpn_rois_bv, rpn_rois_3d


    @layer
    def proposal_target_layer_3d(self, input, classes, name):
        if isinstance(input[0], tuple):
            input_bv = input[0][0]
            # input_img = input[0][1]
            input_3d = input[0][3]
        with tf.variable_scope(name) as scope:
            # print('dtype',input[0].dtype)
            rois_bv, rois_img, labels,bbox_targets_corners, rois_3d = \
            tf.py_func(proposal_target_layer_py_3d,[input_bv,input_3d,input[1],input[2],input[3],input[4],classes],[tf.float32,tf.float32,tf.int32,tf.float32, tf.float32])

            rois_bv = tf.reshape(rois_bv,[-1,5] , name = 'rois_bv')
            rois_img = tf.reshape(rois_img,[-1,5] , name = 'rois_img')
            rois_3d = tf.reshape(rois_3d,[-1,7] , name = 'rois_3d') # for debug
            labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'labels')
            bbox_targets_corners = tf.convert_to_tensor(bbox_targets_corners, name = 'bbox_targets_corners')

            return rois_bv, rois_img, labels, bbox_targets_corners, rois_3d

    @layer
    def proposal_target_layer(self, input, classes, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:

            rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights = \
             tf.py_func(proposal_target_layer_py,[input[0],input[1],classes],[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])

            rois = tf.reshape(rois,[-1,5] , name = 'rois')
            labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'labels')
            bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'bbox_targets')
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name = 'bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name = 'bbox_outside_weights')
            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


    @layer
    def proposal_transform(self, input, name, target='bv'):
        """ transform 3d propasal to different view """

        assert(target in ('bv', 'img', 'fv'))
        if isinstance(input, tuple):
            input_bv = input[0]
            input_img = input[1]

        if target == 'bv':

            with tf.variable_scope(name) as scope:
                lidar_bv = input_bv
            return lidar_bv

        elif target == 'img':

            with tf.variable_scope(name) as scope:
                image_proposal = input_img
            return image_proposal

        elif target == 'fv':
            # TODO
            return None


    # @layer
    # def reshape_layer(self, input, d, name):
    #     input_shape = tf.shape(input)
    #     if name == 'rpn_cls_prob_reshape':
    #         # input: (1, H, W, Axd)
    #         # transpose: (1, A*d, H, W)
    #         # reshape: (1, d, A*H, W)
    #         # transpose: (1, A*H, W, d)
    #          return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
    #                 int(d),tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),input_shape[2]]),
    #          [0,2,3,1],name=name)
    #     else:
    #          return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
    #                 int(d),tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),input_shape[2]]),[0,2,3,1],name=name)

    @layer
    def reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)

        return tf.reshape(input, 
                            [input_shape[0],
                            input_shape[1],
                            -1,
                            int(d)])

    @layer
    def feature_extrapolating(self, input, scales_base, num_scale_base, num_per_octave, name):
        return feature_extrapolating_op.feature_extrapolating(input,
                              scales_base,
                              num_scale_base,
                              num_per_octave,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(values=inputs, axis=axis, name=name)

    # TODO
    @layer
    def element_wise_mean(self, input):
        return None

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable, regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        else:
            return tf.nn.softmax(input,name=name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)
