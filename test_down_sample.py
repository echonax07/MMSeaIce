#!/usr/bin/env python
# coding: utf-8

# # Quick start guide
# This notebook serves as an example of how to train a simple model using pytorch and the ready-to-train AI4Arctic
# challenge dataset. Initially, a dictionary, 'train_options', is set up with relevant options for both the example
# U-Net Convolutional Neural Network model and the dataloader. Note that the weights of the U-Net will be initialised
# at random and therefore not deterministic - results will vary for every training run. Two lists (dataset.json and
# testset.json) include the names of the scenes relevant to training and testing, where the former can be altered
# if desired. Training data is loaded in parallel using the build-in torch Dataset and Dataloader classes, and
# works by randomly sampling a scene and performing a random crop to extract a patch. Each batch will then be compiled
# of X number of these patches with the patch size in the 'train_options'. An obstacle is different grid resolution
# sizes, which is overcome by upsampling low resolution variables, e.g. AMSR2, ERA5, to match the SAR pixels.
# A number of batches will be prepared in parallel and stored until use, depending on the number of workers (processes)
# spawned (this can be changed in 'num_workers' in 'train_options').
# The model is trained on a fixed number of steps according to the number of batches in an epoch,
# defined by the 'epoch_len' parameter, and will run for a total number of epochs depending on the 'epochs' parameter.
# After each epoch, the model is evaluated. In this example, a random number of scenes are sampled among the training
# scenes (and removed from the list of training scenes) to act as a validation set used for the evaluation.
# The model is evaluated with the metrics, and if the current validation attempt is superior to the previous,
# then the model parameters are stored in the 'best_model' file in the directory.
#
# The models are scored on the three sea ice parameters; Sea Ice Concentration (SIC), Stage of Development (SOD) and
# the Floe size (FLOE) with the $R²$ metric for the SIC, and the weighted F1 metric for the SOD and FLOE. The 3 scores
# are combined into a single metric by taking the weighted average with SIC and SOD being weighted with 2 and the FLOE
# with 1.
#
# Finally, once you are ready to test your model on the test scenes (without reference data), the 'test_upload'
# notebook will produce model outputs with your model of choice and save the output as a netCDF file, which can be
# uploaded to the AI4EO.eu website. The model outputs will be evaluated and then you will receive a score.
#
# This quick start notebook is by no means necessary to utilize, and you are more than welcome to develop your own
# data pipeline. We do however require that the model output is stored in a netcdf file with xarray.dataarrays titled
# '{scene_name}_{chart}', i.e. 3 charts per scene / file (see how in 'test_upload'). In addition, you are more than
# welcome to create your own preprocessing scheme to prepare the raw AI4Arctic challenge dataset. However, we ask that
# the model output is in 80 m pixel spacing (original is 40 m), and that you follow the class numberings from the
# lookup tables in 'utils' - at least you will be evaluated in this way. Furthermore, we have included a function to
# convert the polygon_icechart to SIC, SOD and FLOE, you will have to incorporate it yourself.
#
# The first cell imports the necessary Python packages, initializes the 'train_options' dictionary
# the sample U-Net options, loads the dataset list and select validation scenes.
# %%
import argparse
import json
import random
import os
import os.path as osp
import shutil
import time


import numpy as np
import torch
from mmcv import Config, mkdir_or_exist
from tqdm import tqdm  # Progress bar


# Functions to calculate metrics and show the relevant chart colorbar.
from functions import compute_metrics, save_best_model
# Custom dataloaders for regular training and validation.
from loaders import (AI4ArcticChallengeDataset, AI4ArcticChallengeTestDataset,
                     get_variable_options)
#  get_variable_options
from unet import UNet  # Convolutional Neural Network model
# -- Built-in modules -- #
from utils import colour_str

from test_upload_function import test





def create_train_and_validation_scene_list(train_options):
    '''
    Creates the a train and validation scene list. Adds these two list to the config file train_options

    '''
    with open(train_options['path_to_env'] + 'datalists/dataset.json') as file:
        train_options['train_list'] = json.loads(file.read())

    # Convert the original scene names to the preprocessed names.
    train_options['train_list'] = [file[17:32] + '_' + file[77:80] +
                                   '_prep.nc' for file in train_options['train_list']]

    # # Select a random number of validation scenes with the same seed. Feel free to change the seed.et
    # # np.random.seed(0)
    # train_options['validate_list'] = np.random.choice(np.array(
    #     train_options['train_list']), size=train_options['num_val_scenes'], replace=False)

    # load validation list
    with open(train_options['path_to_env'] + train_options['val_path']) as file:
        train_options['validate_list'] = json.loads(file.read())
    # Convert the original scene names to the preprocessed names.
    train_options['validate_list'] = [file[17:32] + '_' + file[77:80] +
                                      '_prep.nc' for file in train_options['validate_list']]

    # from icecream import ic
    # ic(train_options['validate_list'])
    # Remove the validation scenes from the train list.
    train_options['train_list'] = [scene for scene in train_options['train_list']
                                if scene not in train_options['validate_list']]
    print('Options initialised')


def create_dataloaders(train_options):
    '''
    Create train and validation dataloader based on the train and validation list inside train_options.

    '''
    # Custom dataset and dataloader.
    dataset = AI4ArcticChallengeDataset(
        files=train_options['train_list'], options=train_options)

    dataloader_train = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=True, num_workers=train_options['num_workers'], pin_memory=True)
    # - Setup of the validation dataset/dataloader. The same is used for model testing in 'test_upload.ipynb'.

    dataset_val = AI4ArcticChallengeTestDataset(
        options=train_options, files=train_options['validate_list'])

    # dataset_val = AI4ArcticChallengeDataset(
    #     files=train_options['validate_list'], options=train_options)

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)

    return dataloader_train, dataloader_val


cfg = Config.fromfile("configs/Unite_test/test_down_sample.py")
train_options = cfg.train_options
# Get options for variables, amsrenv grid, cropping and upsampling.
train_options = get_variable_options(train_options)
# cfg['experiment_name']=
# cfg.env_dict = {}

# set seed for everything
seed = train_options['seed']
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


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
optimizer = torch.optim.Adam(list(net.parameters()), lr=train_options['lr'])

create_train_and_validation_scene_list(train_options)

dataloader_train, dataloader_val = create_dataloaders(train_options)
#%%

train_features, train_labels = next(iter(dataloader_train))
val_features, val_labels, masks, name, orign_dfs = next(iter(dataloader_val))

# %%
