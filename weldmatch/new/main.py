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
import os
import sys
sys.path.append(os.getcwd()+'/tools')

import keras
import tensorflow as tf
import functools

from model_start import models_backbone
from model_start import load_model
from tools import anchors

import tools.check_keras as check
import tools.retinanet as renet
from tools.freeze_model import freeze
import tools.losses

from tools.eval import Evaluate
from tools.callback_common import RedirectModel

from data_generate import create_generators
from keras.callbacks import EarlyStopping
# os.environ['CUDA_VISIBLE_DEVICES'] =''

def makedirs(path):
	# Intended behavior: try to create the directory,
	# pass if the directory exists already, fails otherwise.
	# Meant for Python 2.7/3.n compatibility.
	try:
		os.makedirs(path)
	except OSError:
		if not os.path.isdir(path):
			raise


def model_with_weights(model, weights, skip_mismatch):
	""" Load weights for model.

	Args
		model         : The model to load weights for.
		weights       : The weights to load.
		skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
	"""
	if weights is not None:
		model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
	return model



def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0, freeze_backbone=False):
	""" Creates three models (model, training_model, prediction_model).

	Args
		backbone_retinanet : A function to call to create a retinanet model with a given backbone.
		num_classes        : The number of classes to train.
		weights            : The weights to load into the model.
		multi_gpu          : The number of GPUs to use for training.
		freeze_backbone    : If True, disables learning for the backbone.

	Returns
		model            : The base model. This is also the model that is saved in snapshots.
		training_model   : The training model. If multi_gpu=0, this is identical to model.
		prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
	"""
	modifier = freeze if freeze_backbone else None

	# Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
	# optionally wrap in a parallel model
	if multi_gpu > 1:
		with tf.device('/cpu:0'):
			model = model_with_weights(backbone_retinanet(num_classes, modifier=modifier), weights=weights, skip_mismatch=True)
		training_model = multi_gpu_model(model, gpus=multi_gpu)
	else:
		model          = model_with_weights(backbone_retinanet(num_classes, modifier=modifier), weights=weights, skip_mismatch=True)
		training_model = model

	# make prediction model
	prediction_model = renet.retinanet_bbox(model=model)

	# compile model
	training_model.compile(
		loss={
			'regression'    : tools.losses.smooth_l1(),
			'classification': tools.losses.focal()
		},
		optimizer=keras.optimizers.adam(lr=1e-4, clipnorm=0.001)
	)

	return model, training_model, prediction_model



def create_callbacks(model, training_model, prediction_model, validation_generator, tensorboard_dir,batch_size,snapshot_path,backbone_name):
	""" Creates the callbacks to use during training.

	Args
		model: The base model.
		training_model: The model that is used for training.
		prediction_model: The model that should be used for validation.
		validation_generator: The generator for creating validation data.
		args: parseargs args object.

	Returns:
		A list of callbacks used for training.
	"""
	callbacks = []
	evaluation = 1
	snapshots = 1 #if save every epoch snapshot
	tensorboard_callback = None

	if tensorboard_dir:
		tensorboard_callback = keras.callbacks.TensorBoard(
			log_dir                = tensorboard_dir,
			histogram_freq         = 0,
			batch_size             = batch_size,
			write_graph            = True,
			write_grads            = False,
			write_images           = False,
			embeddings_freq        = 0,
			embeddings_layer_names = None,
			embeddings_metadata    = None
		)
		callbacks.append(tensorboard_callback)

	if evaluation and validation_generator:
		evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback)
		evaluation = RedirectModel(evaluation, prediction_model)
		callbacks.append(evaluation)

	# save the model
	if snapshots:
		# ensure directory created first; otherwise h5py will error after epoch.
		makedirs(snapshot_path)
		checkpoint = keras.callbacks.ModelCheckpoint(
			os.path.join(
				snapshot_path,
				'{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone='simnet2', dataset_type='res101')
			),
			verbose=1,
			# save_best_only=True,
			# monitor="mAP",
			# mode='max'
		)
		checkpoint = RedirectModel(checkpoint, model)
		callbacks.append(checkpoint)

	callbacks.append(keras.callbacks.ReduceLROnPlateau(
		monitor  = 'loss',
		factor   = 0.1,
		patience = 3,
		verbose  = 1,
		mode     = 'auto',
		epsilon  = 0.0001,
		cooldown = 0,
		min_lr   = 0
	))
	early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=50, verbose=1)
	callbacks.append(early_stopping)
	return callbacks



def get_session():
	""" Construct a modified tf session.
	"""
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	return tf.Session(config=config)


def main(snapshot_path,snapshot = None,backbone_name ='resnet101',tensorboard_dir = None,
		 batch_size=4,steps = 300,epochs = 200,imagenet_weights=True):


	annotations_path = 'Train_file/annotations.csv'
	classes_path = 'Train_file/classes.csv'

	# create object that stores backbone information
	backbone = models_backbone(backbone_name)

	# make sure keras is the minimum required version
	check.check_keras_version()

	#  specific GPU
	'''
	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	'''
	keras.backend.tensorflow_backend.set_session(get_session())

	# create the generators
	train_generator, validation_generator = create_generators(annotations = annotations_path,classes = classes_path,batch_size=batch_size)

	# create the model
	if snapshot is not None:
		print('Loading model, this may take a second...')
		model            = load_model(snapshot, backbone_name=backbone_name)
		training_model   = model
		prediction_model = renet.retinanet_bbox(model=model)
	else:
		# default to imagenet if nothing else is specified
		if imagenet_weights:
			weights = backbone.download_imagenet()
			print(weights)

		print('Creating model, this may take a second...')
		model, training_model, prediction_model = create_models(
			backbone_retinanet=backbone.retinanet,
			num_classes=train_generator.num_classes(),
			weights=weights,
			multi_gpu=0,#default
			freeze_backbone=0#default
		)

	# print model summary
	print(model.summary())
	# return

	# this lets the generator compute backbone layer shapes using the actual backbone model
	if 'vgg' in backbone_name or 'densenet' in backbone_name:
		compute_anchor_targets = functools.partial(anchors.anchor_targets_bbox, shapes_callback=anchors.make_shapes_callback(model))
		train_generator.compute_anchor_targets = compute_anchor_targets
		if validation_generator is not None:
			validation_generator.compute_anchor_targets = compute_anchor_targets

	# create the callbacks
	callbacks = create_callbacks(
		model,
		training_model,
		prediction_model,
		validation_generator,
		tensorboard_dir,
		batch_size,
		snapshot_path,
		backbone_name
	)

	# start training
	training_model.fit_generator(
		generator=train_generator,
		steps_per_epoch= steps,
		epochs= epochs,
		verbose=1,
		callbacks=callbacks,
	)


if __name__ == '__main__':
	main('model_save')

