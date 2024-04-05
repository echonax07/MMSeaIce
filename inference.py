__author__ = 'Muhammed Patel'
__contributor__ = 'Xinwwei chen, Fernando Pena Cantu,Javier Turnes, Eddie Park'
__copyright__ = ['university of waterloo']
__contact__ = ['m32patel@uwaterloo.ca', 'xinweic@uwaterloo.ca']
__version__ = '1.0.0'
__date__ = '2024-04-05'

# -- Built-in modules -- #
import argparse
import json
import random
import os
import os.path as osp
import shutil
from icecream import ic
import pathlib

import numpy as np
import torch
from mmcv import Config, mkdir_or_exist
from tqdm import tqdm  # Progress bar

import wandb
# Functions to calculate metrics and show the relevant chart colorbar.
from functions import compute_metrics, save_best_model, load_model, slide_inference, \
    batched_slide_inference, water_edge_metric, class_decider

# Load consutme loss function
from losses import WaterConsistencyLoss
# Custom dataloaders for regular training and validation.
from loaders import (AI4ArcticChallengeDataset, AI4ArcticChallengeTestDataset,
                     get_variable_options)
#  get_variable_options
from unet import UNet, Sep_feat_dif_stages  # Convolutional Neural Network model
from swin_transformer import SwinTransformer  # Swin Transformer
# -- Built-in modules -- #
from utils import colour_str
from test_upload_function import test
import segmentation_models_pytorch as smp


def parse_args():
    parser = argparse.ArgumentParser(description='Train Default U-NET segmentor')

    # Mandatory arguments
    parser.add_argument('config', type=pathlib.Path, help='train config file path',)
    parser.add_argument('checkpoint', type=pathlib.Path, help='checkpoint path of the model',)
    parser.add_argument('--wandb-project', required=True, help='Name of wandb project')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    ic(args.config)
    cfg = Config.fromfile(args.config)
    train_options = cfg.train_options
    # Get options for variables, amsrenv grid, cropping and upsampling.
    train_options = get_variable_options(train_options)
    # generate wandb run id, to be used to link the run with test_upload
    id = wandb.util.generate_id()

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        if not train_options['cross_val_run']:
            cfg.work_dir = osp.join('./work_dir',
                                    osp.splitext(osp.basename(args.config))[0])
        else:
            # from utils import run_names
            run_name = id
            cfg.work_dir = osp.join('./work_dir',
                                    osp.splitext(osp.basename(args.config))[0], run_name)

    ic(cfg.work_dir)
    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    cfg_path = osp.join(cfg.work_dir, osp.basename(args.config))
    # ### CUDA / GPU Setup
    # Get GPU resources.
    if torch.cuda.is_available():
        print(colour_str('GPU available!', 'green'))
        print('Total number of available devices: ',
              colour_str(torch.cuda.device_count(), 'orange'))
        device = torch.device(f"cuda:{train_options['gpu_id']}")

    else:
        print(colour_str('GPU not available.', 'red'))
        device = torch.device('cpu')
    print('GPU setup completed!')

    if train_options['model_selection'] == 'unet':
        net = UNet(options=train_options).to(device)
    elif train_options['model_selection'] == 'swin':
        net = SwinTransformer(options=train_options).to(device)
    elif train_options['model_selection'] == 'h_unet':
        from unet import H_UNet
        net = H_UNet(options=train_options).to(device)
    elif train_options['model_selection'] == 'h_unet_argmax':
        from unet import H_UNet_argmax
        net = H_UNet_argmax(options=train_options).to(device)
    elif train_options['model_selection'] == 'Separate_decoder':
        net = Sep_feat_dif_stages(options=train_options).to(device)
    elif train_options['model_selection'] in ['UNet_regression', 'unet_regression']:
        from unet import UNet_regression
        net = UNet_regression(options=train_options).to(device)
    elif train_options['model_selection'] in ['UNet_regression_all']:
        from unet import UNet_regression_all
        net = UNet_regression_all(options=train_options).to(device)
    elif train_options['model_selection'] in ['UNet_sep_dec_regression', 'unet_sep_dec_regression']:
        from unet import UNet_sep_dec_regression
        net = UNet_sep_dec_regression(options=train_options).to(device)
    elif train_options['model_selection'] in ['UNet_sep_dec_mse']:
        from unet import UNet_sep_dec_mse
        net = UNet_sep_dec_mse(options=train_options).to(device)
    else:
        raise 'Unknown model selected'

    wandb.init(name=osp.splitext(osp.basename(args.config))[0] + '_inference', project=args.wandb_project,
               entity="ai4arctic", config=train_options, id=id, resume="allow")

    test(False, net, checkpoint_path, device, cfg, train_options['test_path_gt_embedded_json'])

    # test(False, net, checkpoint_path, device, cfg, train_options['val_path'])
    # todo
    # this is for valset 2 visualization along with gt
    # test(False, net, checkpoint_path, device, cfg, train_options['test_path'])

    # finish the wandb run
    wandb.finish()


if __name__ == '__main__':
    main()
