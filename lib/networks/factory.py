# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import networks.VGGnet_train
import networks.VGGnet_test
import pdb
import tensorflow as tf
from networks.MV3D_train import MV3D_train
from networks.MV3D_test import MV3D_test
#__sets['VGGnet_train'] = networks.VGGnet_train()

#__sets['VGGnet_test'] = networks.VGGnet_test()


def get_network(name):
    """Get a network by name."""
    #if not __sets.has_key(name):
    #    raise KeyError('Unknown dataset: {}'.format(name))
    #return __sets[name]
    if name.split('_')[1] == 'test':
       return MV3D_test()
    elif name.split('_')[1] == 'train':
       return MV3D_train()
    else:
       raise KeyError('Unknown dataset: {}'.format(name))


def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
