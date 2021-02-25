#python3
# -*- coding: utf-8 -*-
"""
Created on 2018/9/5 18:01
@author  : xfby
@Software: PyCharm

"""

'''
give datas

'''

from tools import transform
from tools import csv_generate

def create_generators(annotations,classes,batch_size,random_transform = True,val_annotations = None,dataset_type='csv'):
    """ Create generators for training and validation.
    """
    # create random transform generator for augmenting training data
    if random_transform:
        transform_generator = transform.random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0,
            flip_y_chance=0,
        )
    else:
        transform_generator = transform.random_transform_generator(flip_x_chance=0.5)

    
    if dataset_type == 'csv':
        train_generator = csv_generate.CSVGenerator(
            annotations,
            classes,
            transform_generator=transform_generator,
            batch_size=batch_size,
            image_min_side= 800, #800,#change you can
            image_max_side= 1500 #1333  512  800
        )

        if val_annotations:
            validation_generator = csv_generate.CSVGenerator(
                val_annotations,
                classes,
                batch_size=batch_size,
                image_min_side=800,  # 800,#change you can
                image_max_side=1000  # 1333
            )
        else:
            validation_generator = None
    
    else:
        raise ValueError('Invalid data type received: {}'.format('except:csv---please transfer it !'))

    return train_generator, validation_generator
