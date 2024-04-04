#!/usr/bin/env python
# -*-coding:utf-8 -*-


_base_ = ['../_base_/base2.py']

# dual pole with incedence angle distance map and environment variables
SCENE_VARIABLES = [
    # -- Sentinel-1 variables -- #
    'nersc_sar_primary',
    'nersc_sar_secondary',
    'sar_incidenceangle',

    # -- Geographical variables -- #
    'distance_map',

    # -- AMSR2 channels -- #
    'btemp_6_9h', 'btemp_6_9v',
    # 'btemp_7_3h', 'btemp_7_3v',
    # 'btemp_10_7h', 'btemp_10_7v',
    # 'btemp_18_7h', 'btemp_18_7v',
    'btemp_23_8h', 'btemp_23_8v',
    # 'btemp_36_5h', 'btemp_36_5v',
    'btemp_89_0h', 'btemp_89_0v',

    # -- Environmental variables -- #
    # 'u10m_rotated', 'v10m_rotated',
    't2m', 'skt', 'tcwv', 'tclw',

    # # -- Auxilary Variables -- #
    # 'aux_time',
    'aux_lat',
    'aux_long'
]

train_options = {

    'train_variables': SCENE_VARIABLES,
    'patch_size': 256,
    'down_sample_scale': 5,
    'unet_conv_filters': [32, 64, 128, 128],

    #-------- The following variables will never change in this experiment run ------#
    'compute_classwise_f1score': True,
    'seed': 10,

    'optimizer': {
        'type': 'SGD',
        'lr': 0.001,  # Optimizer learning rate.
        'momentum': 0.9,
        'dampening': 0,
        'nesterov': False,
        'weight_decay': 0.01
    },

    'scheduler': {
        '_delete_': True,
        'type': 'CosineAnnealingLR',  # Name of the schedulers
        'lr_min': 0,  # Minimun learning rate
    },

    'batch_size': 16,
    'num_workers': 4,  # Number of parallel processes to fetch data.
    'num_workers_val': 4,  # Number of parallel processes during validation.
}
