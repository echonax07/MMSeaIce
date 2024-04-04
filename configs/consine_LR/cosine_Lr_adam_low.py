#!/usr/bin/env python
# -*-coding:utf-8 -*-




_base_ = ['../_base_/base.py']

SCENE_VARIABLES = [
    # -- Sentinel-1 variables -- #
    'nersc_sar_primary',
    'nersc_sar_secondary',
    'sar_incidenceangle',

    # -- Geographical variables -- #
    'distance_map',

    # -- AMSR2 channels -- #
    # 'btemp_6_9h', 'btemp_6_9v',
    # 'btemp_7_3h', 'btemp_7_3v',
    # 'btemp_10_7h', 'btemp_10_7v',
    'btemp_18_7h', 'btemp_18_7v',
    # 'btemp_23_8h', 'btemp_23_8v',
    'btemp_36_5h', 'btemp_36_5v',
    # 'btemp_89_0h', 'btemp_89_0v',

    # -- Environmental variables -- #
    'u10m_rotated', 'v10m_rotated',
    't2m', 
    # 'skt',
    'tcwv', 'tclw',

       # -- Auxilary Variables -- #
    'aux_time',
    'aux_lat',
    'aux_long'
]


train_options = {'train_variables': SCENE_VARIABLES,
                 'val_path': 'datalists/test_val.json',
                 'path_to_train_data': '../../dataset/train',
                 'path_to_test_data': '../../dataset/test',

                 'optimizer': {
                    'type': 'Adam',
                    'lr': 0.00001,  # Optimizer learning rate.
                    'lr_min': 0.0,  # Optimizer learning rate.
                    'b1': 0.9, 
                    'b2': 0.999,
                    'weight_decay': 0.0
                },
                 'scheduler': 'CosineAnnealingLR',
                 'batch_size': 16,
                 'num_workers': 12,  # Number of parallel processes to fetch data.
                 'num_workers_val': 12,  # Number of parallel processes during validation.
                 'patch_size': 256,
                 'down_sample_scale': 10,
}