#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   base.py
@Time    :   2023/01/26 15:18:42
@Author  :   Muhammed Patel
@Version :   1.0
@Contact :   m32patel@uwaterloo.ca
@License :   (C)Copyright 2022-2023, VIP Lab
@Desc    :   None
'''
from functions import f1_metric, r2_metric
import torch.nn as nn
# Charts in the dataset
CHARTS = ['SIC', 'SOD', 'FLOE']


SCENE_VARIABLES = [
    # -- Sentinel-1 variables -- #
    'nersc_sar_primary',
    'nersc_sar_secondary',
    'sar_incidenceangle',

    # -- Geographical variables -- #
    'distance_map',

    # -- AMSR2 channels -- #
    'btemp_6_9h', 'btemp_6_9v',
    'btemp_7_3h', 'btemp_7_3v',
    'btemp_10_7h', 'btemp_10_7v',
    'btemp_18_7h', 'btemp_18_7v',
    'btemp_23_8h', 'btemp_23_8v',
    'btemp_36_5h', 'btemp_36_5v',
    'btemp_89_0h', 'btemp_89_0v',

    # -- Environmental variables -- #
    'u10m_rotated', 'v10m_rotated',
    't2m', 'skt', 'tcwv', 'tclw',

    # -- Auxilary Variables -- #
    'aux_time',
    'aux_lat',
    'aux_long'
]

# Sea Ice Concentration (SIC) code to class conversion lookup table.
SIC_LOOKUP = {
    'polygon_idx': 0,  # Index of polygon number.
    'total_sic_idx': 1,  # Total Sea Ice Concentration Index, CT.
    'sic_partial_idx': [2, 5, 8],  # Partial SIC polygon code index. CA, CB, CC.
    0: 0,
    1: 0,
    2: 0,
    55: 0,
    10: 1,  # 10 %
    20: 2,  # 20 %
    30: 3,  # 30 %
    40: 4,  # 40 %
    50: 5,  # 50 %
    60: 6,  # 60 %
    70: 7,  # 70 %
    80: 8,  # 80 %
    90: 9,  # 90 %
    91: 10,  # 100 %
    92: 10,  # Fast ice
    'mask': 255,
    'n_classes': 12
}

# Stage of Development code to class conversion lookup table.
SOD_LOOKUP = {
    'sod_partial_idx': [3, 6, 9],  # Partial SIC polygon code index. SA, SB, SC.
    'threshold': 0.7,  # < 1. Minimum partial percentage SIC of total SIC to select SOD. Otherwise ambiguous polygon.
                       # larger than threshold.
    'invalid': -9,  # Value for polygons where the SOD is ambiguous or not filled.
    'water': 0,
    0: 0,
    80: 0,  # No stage of development
    81: 1,  # New ice
    82: 1,  # Nilas, ring ice
    83: 2,  # Young ice
    84: 2,  # Grey ice
    85: 2,  # White ice
    86: 4,  # First-year ice, overall categary
    87: 3,  # Thin first-year ice
    88: 3,  # Thin first-year ice, stage 1
    89: 3,  # Thin first-year ice, stage 2
    91: 4,  # Medium first-year ice
    93: 4,  # Thick first-year ice
    95: 5,  # Old ice
    96: 5,  # Second year ice
    97: 5,  # Multi-year ice
    98: 255,  # Glacier ice
    99: 255,
    'mask': 255,
    'n_classes': 7
}

# Ice floe/form code to class conversion lookup table.
FLOE_LOOKUP = {
    'floe_partial_idx': [4, 7, 10],  # Partial SIC polygon code index. FA, FB, FC.
    'threshold': 0.5,  # < 1. Minimum partial concentration to select floe. Otherwise polygon may be ambiguous.
    'invalid': -9,  # Value for polygons where the floe is ambiguous or not filled.
    'water': 0,
    0: 0,
    22: 255,  # Pancake ice
    1: 255,  # Shuga / small ice cake
    2: 1,  # Ice cake
    3: 2,  # Small floe
    4: 3,  # Medium floe
    5: 4,  # Big floe
    6: 5,  # Vast flpe
    7: 5,  # Gian floe
    8: 255,  # Fast ice
    9: 6,  # Growlers, floebergs or floebits
    10: 6,  # Icebergs
    21: 255,  # Level ice
    'fastice_class': 255,
    'mask': 255,
    'n_classes': 8
}


train_options = {
    # -- Random Seed -- #
    'seed': -1,
    # -- Training options -- #
    # Replace with data directory path.
    'path_to_train_data': '../../dataset/train',
    'path_to_test_data': '../../dataset/test',
    # Which train set is going to be used
    'train_list_path': 'datalists/dataset.json',
    # Which validation set is going to be used
    'val_path': 'datalists/valset2.json',
    # Which test set is going to be used

    'test_path': 'datalists/dataset_test_gt_embedded.json',

    'path_to_env': './',

    'epochs': 100,  # Number of epochs before training stop.
    'epoch_len': 500,  # Number of batches for each epoch.
    # Size of patches sampled. Used for both Width and Height.
    'patch_size': 256,
    'batch_size': 8,  # Number of patches for each batch.
    # How to upscale low resolution variables to high resolution.
    'loader_upsampling': 'nearest',
    # How to down scale low resolution variables to low resolution.
    'loader_downsampling': 'nearest',
    # Down Sampling scale (If it is by 2 the image will get downsample by 2)
    'down_sample_scale': 1,
    'task_weights': [1, 1, 1],
    
    # p leave out cross val run
    'cross_val_run': False,
    'p-out': 20,
    'compute_classwise_f1score': False,
    'plot_confusion_matrix': False,

    # If true it compiles the model. This will increase the speed of the model for training and inference. 
    # Needs Pytorch 2 to work.
    'compile_model': False,
    
    # ensemble after softmax?
    # 'ensemble_after_softmax': True,
    # TODO: Speak with Muhammed different quickstart.py should have different configs.

    # -- loss options -- #
    'chart_loss': {  # Loss for the task
        'SIC': {
            'type': 'CrossEntropyLoss',
            'ignore_index': 255,
        },
        'SOD': {
            'type': 'CrossEntropyLoss',
            'ignore_index': 255,
        },
        'FLOE': {
            'type': 'CrossEntropyLoss',
            'ignore_index': 255,

        },
    },
    # Having the following paramter greater than zero adds a new loss that penilizies inconsitent
    # water logits across the three outputs (SIC, SOD, FLOE)
    "edge_consistency_loss": 0,
    # refer this link for more parameters related to loss :
    # https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/losses

    # some losses are defined locally in the losses.py file, refer them to see more
    # about the parameters required


    # # Binary cross entropy loss
    #     'type': 'BinaryCrossEntropyLoss',

    # # Focal Loss
    #     'type': 'FocalLoss',
    #     'mode': 'multiclass',
    #     'gamma': 0.7,
    #     'ignore_index': 255

    # # Dice Loss
    #     'type': 'DiceLoss', --> Not working
    #     'mode': 'multiclass',
    #     'ignore_index': 255

    # # Jaccard Loss --> Not working
    #     'type': 'JaccardLoss',
    #     'mode': 'multiclass',

    # # LovaszLoss Loss
    #     'type': 'LovaszLoss',
    #     'mode': 'multiclass',
    #     'ignore_index': 255

    # # SoftBCEWithLogitsLoss Loss
    #     'type': 'SoftBCEWithLogitsLoss',
    #     'ignore_index': 255,
    #     'smooth_factor': 0,

    # # SoftCrossEntropyLoss --> NOT WORKING
    #     'type': 'SoftCrossEntropyLoss',
    #     'ignore_index': 255,
    #     'smooth_factor': 0.1,

    # # TverskyLoss
    #     'type': 'TverskyLoss',
    #     'ignore_index': 255,
    #     'mode': 'multiclass',
    #     'smooth': 0,

    # # MCC Loss --> NOT WORKING
    #     'type': 'MCCLoss',
    #     'eps': 1e-5

    # CrossEntropyLoss from Pytorch
    #     'type': 'CrossEntropyLoss',
    #     'ignore_index': 255,

    # # BinaryCrossEntropyLoss from Pytorch --> Not working
    #     'type': 'BCELoss',
    #     'ignore_index': 255,
    #     #  'smooth_factor': 0.1,

    # OrderedCrossEntropyLoss
    #     'type': 'OrderedCrossEntropyLoss',
    #     'ignore_index': 255,

    # MSELossFromLogits
    #     'type': 'MSELossFromLogits',
    #     'ignore_index': 255,

    # If true accuracy will be calculated using binary classification between water and ice first
    "binary_water_classifier": False,

    'optimizer': {
        'type': 'Adam',
        'lr': 0.0001,  # Optimizer learning rate.
        'b1': 0.9,
        'b2': 0.999,
        'weight_decay': 0.0
    },

    # 'optimizer': {
    #     'type': 'AdamW',
    #     'lr': 0.0001,  # Optimizer learning rate.
    #     'b1': 0.9,
    #     'b2': 0.999,
    #     'weight_decay': 0.01
    # },

    # 'optimizer': {
    #     'type': 'SGD',
    #     'lr': 0.0001,  # Optimizer learning rate.
    #     'momentum': 0,
    #     'dampening': 0,
    #     'nesterov': False,
    #     'weight_decay': 0.01
    # },

    'scheduler': {'type': None},

    # 'scheduler': {
    # '_delete_': True,
    #     'type': 'CosineAnnealingLR',  # Name of the schedulers
    #     'lr_min': 0,  # Minimun learning rate
    # },

    # 'scheduler': {
    # '_delete_': True,
    #     'type': 'CosineAnnealingWarmRestartsLR',  # Name of the schedulers
    #     'EpochsPerRestart': 20,  # Number of epochs for the first restart
    #     'RestartMult': 1,  # This number will be used to increase or descrase the number of epochs to restart after each restart.
    #     'lr_min': 0,  # Minimun learning rate
    # },

    # -- Data prepraration lookups and metrics.
    # Contains the relevant variables in the scenes.
    'train_variables': SCENE_VARIABLES,
    'charts': CHARTS,  # Charts to train on.
    'n_classes': {  # number of total classes in the reference charts, including the mask.
        'SIC': SIC_LOOKUP['n_classes'],
        'SOD': SOD_LOOKUP['n_classes'],
        'FLOE': FLOE_LOOKUP['n_classes']
    },
    # SAR pixel spacing. 80 for the ready-to-train AI4Arctic Challenge dataset.
    'pixel_spacing': 80,
    'train_fill_value': 0,  # Mask value for SAR training data.
    'class_fill_values': {  # Mask value for class/reference data.
        'SIC': SIC_LOOKUP['mask'],
        'SOD': SOD_LOOKUP['mask'],
        'FLOE': FLOE_LOOKUP['mask'],
    },

    # -- Validation options -- #
    'chart_metric': {  # Metric functions for each ice parameter and the associated weight.
        'SIC': {
            'func': r2_metric,
            'weight': 2,
        },
        'SOD': {
            'func': f1_metric,
            'weight': 2,
        },
        'FLOE': {
            'func': f1_metric,
            'weight': 1,
        },
    },
    # Number of scenes randomly sampled from train_list to use in validation.
    'num_val_scenes': 10,

    # -- GPU/cuda options -- #
    'gpu_id': 0,  # Index of GPU. In case of multiple GPUs.
    'num_workers': 0,  # Number of parallel processes to fetch data.
    'num_workers_val': 0,  # Number of parallel processes during validation.

    # -- U-Net Options -- #
    'unet_conv_filters': [16, 32, 64, 64],  # Number of filters in the U-Net.
    'conv_kernel_size': (3, 3),  # Size of convolutional kernels.
    'conv_stride_rate': (1, 1),  # Stride rate of convolutional kernels.
    'conv_dilation_rate': (1, 1),  # Dilation rate of convolutional kernels.
    'conv_padding': (1, 1),  # Number of padded pixels in convolutional layers.
    'conv_padding_style': 'zeros',  # Style of padding.

    # -- Swin Transformer Options -- #
    'swin_hp': {'patch_size': 4,  # (int | tuple(int)): Patch size. Default: 4
                'embed_dim': 96,  # (int): Patch embedding dimension. Default: 96
                'depths': [2, 2, 6, 2],  # (tuple(int)): Depth of each Swin Transformer layer.
                'num_heads': [3, 6, 12, 24],  # (tuple(int)): Number of attention heads in different layers.
                'window_size': 8,  # (int): Window size. Default: 8
                'mlp_ratio': 4.,  # (float): Ratio of mlp hidden dim to embedding dim. Default: 4
                'qkv_bias': True,  # (bool): If True, add a learnable bias to query, key, value. Default: True
                'qk_scale': None,  # (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
                'drop_rate': 0.,  # (float): Dropout rate. Default: 0
                'attn_drop_rate': 0.,  # (float): Attention dropout rate. Default: 0
                'drop_path_rate': 0.1,  # (float): Stochastic depth rate. Default: 0.1
                'norm_layer': nn.LayerNorm,  # (nn.Module): Normalization layer. Default: nn.LayerNorm.
                'ape': False,  # (bool): If True, add absolute position embedding to the patch embedding. Default: False
                'patch_norm': True,  # (bool): If True, add normalization after patch embedding. Default: True
                'use_checkpoint': False,  # (bool): Whether to use checkpointing to save memory. Default: False
                # (tuple(int)): Stride for the sliding inference method during validation. Default 56
                'val_stride': [56, 56],
                # (tuple(int)): Stride for the sliding inference method during testing. Default 4
                'test_stride': [4, 4],
                },

    # -- Separate_decoder options -- #
    'common_features_last_layer': 6,
    'backbone': 'unet',

    # -- Model selection -- #
    'model_selection': 'unet',  # Either 'swin', 'unet, or Separate_decoder

    # -- Latitude and Longitude Information for Normalization -- #
    'latitude': {
        'mean': 69.12526250065734,
        'std': 7.03179625261593
    },

    'longitude': {
        'mean': -56.38966259295485,
        'std': 31.32935694114249
    },

    'amsrenv_pixel_spacing': 2000,  # amsrenv pixel spacing is 2 km


    # -- Data Augmentation Options -- #
    # Random_h_flip (float, [0, 1]) - Chance of doing random horizontal flip
    # Random_v_flip (float, [0, 1]) - Chance of doing random vertical flip
    # Random_rotation_prob (float, [0, 1]) - Chance of doing random rotation
    # Random_rotation (int, [0, 180]) - Maximum random degree rotation between 0 and Random_rotation
    # Random_scale_prob (float, [0, 1]) - Chance of doing random scaling
    # Random_scale (float_tuple(a, b)) - Scales image randomly to a scale between a and b. b > a.
    # Cutmix_beta (float, (0, inf)) - Parameter for random beta distribution
    # Cutmix_prob (float, [0, 1]) - Chance of doing random cutmix
    'data_augmentations': {
        'Random_h_flip': 0,
        'Random_v_flip': 0,
        'Random_rotation_prob': 0,
        'Random_rotation': 0,
        'Random_scale_prob': 0,
        'Random_scale': (1, 1),
        'Cutmix_beta': 1.0,
        'Cutmix_prob': 0,
    }
}
