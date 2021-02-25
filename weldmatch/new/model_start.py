#python3
# -*- coding: utf-8 -*-
"""
Created on 2018/9/4 上午8:16 
@author  : xfby
@Software: PyCharm

"""

'''
pass

'''

import keras
from tools import model_2d
from tools import misc
from  tools import initializers
from tools import filter_detection
from tools import losses
from tools import layer_bn
from tools import retinanet

custom_objects = {
            'UpsampleLike'     : misc.UpsampleLike,
            'PriorProbability' : initializers.PriorProbability,
            'RegressBoxes'     : misc.RegressBoxes,
            'FilterDetections' : filter_detection.FilterDetections,
            'Anchors'          : misc.Anchors,
            'ClipBoxes'        : misc.ClipBoxes,
            '_smooth_l1'       : losses.smooth_l1(),
            '_focal'           : losses.focal(),
            'BatchNormalization': layer_bn.BatchNormalization
        }

class ResNetBackbone(object):
    """ 
    Describes backbone information and provides utility functions
    
    """

    def __init__(self, backbone):

        self.backbone = backbone
        self._validate()
        
#    @staticmethod
    def retinanet(self,*args, **kwargs):
        """ 
        Returns a retinanet model using the correct backbone.
        """
        return self.resnet_retinanet(*args, backbone=self.backbone, **kwargs)
    
#    @staticmethod
    def download_imagenet(self):
        """ 
        Downloads ImageNet weights and returns path of weights file.
        
        """
        resnet_filename = 'ResNet-{}-model.keras.h5'
        resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)
        depth = int(self.backbone.replace('resnet', ''))

        filename = resnet_filename.format(depth)
        resource = resnet_resource.format(depth)
        if depth == 50:
            checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
        elif depth == 101:
            checksum = '05dc86924389e5b401a9ea0348a3213c'
        elif depth == 152:
            checksum = '6ee11ef2b135592f8031058820bb9e71'

        return keras.applications.imagenet_utils.get_file(
            filename,
            resource,
            cache_subdir='models',
            md5_hash=checksum
        )
    
    def resnet_retinanet(self,num_classes, backbone='resnet50', inputs=None, modifier=None, **kwargs):
        """ Constructs a retinanet model using a resnet backbone.
    
        Args
            num_classes: Number of classes to predict.
            backbone: Which backbone to use (one of ('resnet50', 'resnet101', 'resnet152')).
            inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
            modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).
    
        Returns
            RetinaNet model with a ResNet backbone.
        """
        # choose default input
        if inputs is None:
            inputs = keras.layers.Input(shape=(None, None, 3))
    
        # create the resnet backbone
        if backbone == 'resnet50':
            resnet = model_2d.ResNet50(inputs, include_top=False, freeze_bn=True)
        elif backbone == 'resnet101':
            resnet = model_2d.ResNet101(inputs, include_top=False, freeze_bn=True)
        elif backbone == 'resnet152':
            resnet = model_2d.ResNet152(inputs, include_top=False, freeze_bn=True)
        else:
            raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))
    
        # invoke modifier if given
        if modifier:
            resnet = modifier(resnet)
        # create the full model
        return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=resnet.outputs[1:], **kwargs)    
    
    
    def _validate(self):
        """
        Checks whether the backbone string is correct.
        just resnet50,resnet101,resnet152
        """
        allowed_backbones = ['resnet50', 'resnet101', 'resnet152']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))





def models_backbone(backbone_name):
    if 'resnet' in backbone_name:
        return ResNetBackbone(backbone_name)
        
#    this can implemented other model desnet/vgg/...
        
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone_name))


def load_model(filepath: object, backbone_name: object = 'resnet50', convert: object = False, nms: object = True) -> object:
    """ Loads a retinanet model using the correct custom objects.
    # Arguments
        filepath: one of the following:
            - string, path to the saved model, or
            - h5py.File object from which to load the model
        backbone_name: Backbone with which the model was trained.
        convert: Boolean, whether to convert the model to an inference model.
        nms: Boolean, whether to add NMS filtering to the converted model. Only valid if convert=True.

    # Returns
        A keras.models.Model object.

    # Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    import keras.models

    model = keras.models.load_model(filepath, custom_objects=custom_objects)
    if convert:
        import tools.retinanet as renet
        model = renet.retinanet_bbox(model=model, nms=nms)

    return model

