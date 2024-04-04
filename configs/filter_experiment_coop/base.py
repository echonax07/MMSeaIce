#!/usr/bin/env python
# -*-coding:utf-8 -*-


_base_ = ['../_base_/base2.py']




train_options = {
                 
                'unet_conv_filters': [16, 32, 64, 64],

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
                 'patch_size': 256,
                 'down_sample_scale': 5,
                 }
