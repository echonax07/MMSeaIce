#!/usr/bin/env python
# -*-coding:utf-8 -*-


_base_ = ['../_base_/base2.py']


train_options = {

    # -- loss options -- #
    'chart_loss': {  # Loss for the task
        'SIC': {
            'type': 'FocalLoss',
            'mode': 'multiclass',
            'gamma': 5,
            'ignore_index': 255
        },
        'SOD': {
            'type': 'FocalLoss',
            'mode': 'multiclass',
            'gamma': 5,
            'ignore_index': 255
        },
        'FLOE': {
            'type': 'FocalLoss',
            'mode': 'multiclass',
            'gamma': 5,
            'ignore_index': 255
        },
    },


    'data_augmentations': {
        'Random_h_flip': 0.5,
        'Random_v_flip': 0.5,
        'Random_rotation_prob': 0.5,
        'Random_rotation': 90,
        'Random_scale_prob': 0.5,
        'Random_scale': (0.9, 1.1),
        'Cutmix_beta': 1.0,
        'Cutmix_prob': 0.5,
    },

    'optimizer': {
        'type': 'SGD',
        'lr': 0.001,  # Optimizer learning rate.
        'momentum': 0.9,
        'dampening': 0,
        'nesterov': False,
        'weight_decay': 0.01
    },

    # -------- The following variables will never change in this experiment run ------#

    'compute_classwise_f1score': True,
    'seed': 10,

    'scheduler': {
        '_delete_': True,
        'type': 'CosineAnnealingLR',  # Name of the schedulers
        'lr_min': 0,  # Minimun learning rate
    },

    'batch_size': 16,
    'num_workers': 4,  # Number of parallel processes to fetch data.
    'num_workers_val': 4,  # Number of parallel processes during validation.
    'patch_size': 256,
    'down_sample_scale': 5,

    'unet_conv_filters': [32, 64, 128, 128],
}
