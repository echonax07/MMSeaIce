import argparse
import json
import random
import os
import os.path as osp
import shutil
from icecream import ic

import numpy as np
import torch
from mmcv import Config, mkdir_or_exist
from tqdm import tqdm  # Progress bar

import wandb
# Functions to calculate metrics and show the relevant chart colorbar.
from functions import compute_metrics, save_best_model
# Custom dataloaders for regular training and validation.
from loaders import (AI4ArcticChallengeDataset, AI4ArcticChallengeTestDataset,
                     get_variable_options)
#  get_variable_options
from unet import UNet, UNet_sep_dec  # Convolutional Neural Network model
# -- Built-in modules -- #
from utils import colour_str

from test_upload_function_local import test




cfg = Config.fromfile("work_dir/down_scale_by_5_location/down_scale_by_5_location.py")
checkpoint_path = "work_dir/down_scale_by_5_location/best_model_down_scale_by_5_location.pth"
cfg.work_dir = "work_dir/down_scale_by_5_location"

train_options = cfg.train_options
train_options = get_variable_options(train_options)

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

net = UNet(options=train_options).to(device)

test(net, checkpoint_path, device, cfg)
