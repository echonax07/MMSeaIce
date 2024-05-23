
#!/usr/bin/env python
# coding: utf-8

# Inputs:
#   - config file May not be needed ???
#   - folder path in work_dir

import argparse
import json
import random
import os
import os.path as osp
import glob
import shutil
from icecream import ic
import pathlib
import warnings
import time 
import numpy as np
import torch
from mmcv import Config, mkdir_or_exist
from tqdm import tqdm  # Progress bar

import wandb
# Functions to calculate metrics and show the relevant chart colorbar.
from functions import compute_metrics, save_best_model, load_model, slide_inference, \
    batched_slide_inference, water_edge_metric, class_decider, create_train_validation_and_test_scene_list, \
    get_scheduler, get_optimizer, get_loss, get_model

# Load consutme loss function
from losses import WaterConsistencyLoss
# Custom dataloaders for regular training and validation.
from loaders import get_variable_options, AI4ArcticChallengeDataset, AI4ArcticChallengeTestDataset
#  get_variable_options

# -- Built-in modules -- #
from utils import colour_str
from test_upload_function import test


def parse_args():
    parser = argparse.ArgumentParser(description='Train Default U-NET segmentor')

    # Mandatory arguments
    parser.add_argument('config', type=pathlib.Path, help='train config file path',)
    parser.add_argument('work_dir_folder',type=pathlib.Path, help='folder where all the runs which you want to re run are saved') 
    parser.add_argument('--wandb-project', required=True, help='Name of wandb project')

    args = parser.parse_args()

    return args


def create_dataloaders(train_options):
    '''
    Create train and validation dataloader based on the train and validation list inside train_options.

    '''
    # Custom dataset and dataloader.
    dataset = AI4ArcticChallengeDataset(
        files=train_options['train_list'], options=train_options, do_transform=True)

    dataloader_train = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=True, num_workers=train_options['num_workers'], pin_memory=True)
    # - Setup of the validation dataset/dataloader. The same is used for model testing in 'test_upload.ipynb'.

    dataset_val = AI4ArcticChallengeTestDataset(
        options=train_options, files=train_options['validate_list'], mode='train')

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)

    return dataloader_train, dataloader_val


def check_folder_exists(folder_path):
    return os.path.isdir(folder_path)

def get_file_names(directory_path):
    return os.listdir(directory_path)


def get_checkpoint(directory):

    # use glob to match the pattern '*.pth'
    files = glob.glob(os.path.join(directory, '*.pth'))

    # files is a list of all .pth files. If you're sure there's only one, you can just take the first element
    if files:
        filename = os.path.basename(files[0])
        file_directory = os.path.join(directory,filename)

    else:
        raise FileNotFoundError(f"No .pth file found in the directory {directory}.")

    return file_directory

def create_validation_and_test_scene_list(train_options,run):
    '''
    Creates the a train and validation scene list. Adds these two list to the config file train_options

    '''
    # Validation ---------
    # get all the name of the images in validation
    val_files = get_file_names(os.path.join(train_options['work_dir_folder'],run,'inference_val'))

    val_files = [f[:-4]+'_prep.nc' for f in val_files]
        
    train_options['validate_list'] = val_files



    # Test ----------
    with open(train_options['path_to_env'] + train_options['test_path']) as file:
        train_options['test_list'] = json.loads(file.read())
        train_options['test_list'] = [file[17:32] + '_' + file[77:80] + '_prep.nc'
                                      for file in train_options['test_list']]



def main():
    # start_time = time.time()

    args = parse_args()
    ic(args.config)

    cfg = Config.fromfile(args.config)
    train_options = cfg.train_options

    # Get options for variables, amsrenv grid, cropping and upsampling.
    train_options = get_variable_options(train_options)

    # get work directory folder
    train_options['work_dir_folder'] = args.work_dir_folder

    # ### CUDA / GPU Setup
    # Get GPU resources.
    if torch.cuda.is_available():
        print(colour_str('GPU available!', 'green'))
        print('Total number of available devices: ',
              colour_str(torch.cuda.device_count(), 'orange'))
        
        # Check if NVIDIA V100, A100, or H100 is available for torch compile speed up
        if train_options['compile_model']:
            gpu_ok = False
            device_cap = torch.cuda.get_device_capability()
            if device_cap in ((7, 0), (8, 0), (9, 0)):
                gpu_ok = True
            
            if not gpu_ok:
                warnings.warn(
                    colour_str("GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower than expected.", 'red')
                )

        # Setup device to be used
        device = torch.device(f"cuda:{train_options['gpu_id']}")

    else:
        print(colour_str('GPU not available.', 'red'))
        device = torch.device('cpu')
    print('GPU setup completed!')

    # Loop throught each run
    folders = os.listdir(train_options['work_dir_folder'])

    for run in folders:
        print(colour_str('---------------------------------------------------------------', 'green'))
        print(colour_str(f'Starting {run}:', 'green'))

        #  check if they did validation if not continue
        if not check_folder_exists(os.path.join(train_options['work_dir_folder'],run,'inference_val')):
            continue


        # Create the workdir for the run
        cfg.work_dir = osp.join('./work_dir',
                                    osp.splitext(osp.basename(args.config))[0], run)


        # get all the name of the images in validation and for testing
        create_validation_and_test_scene_list(train_options,run)

        #  load wandb name the run based run we are looping
        wandb.init(name=osp.splitext(osp.basename(args.config))[0]+'-'+run+'-Inference', group=osp.splitext(osp.basename(args.config))[0], project=args.wandb_project,
                   entity="ai4arctic", config=train_options, id=run)

        net = get_model(train_options, device)

        if train_options['compile_model']:
            net = torch.compile(net)

        optimizer = get_optimizer(train_options, net)

        scheduler = get_scheduler(train_options, optimizer)

        # Load Checkpoint
        checkpoint_path = get_checkpoint(os.path.join(train_options['work_dir_folder'],run))

        # Define the metrics and make them such that they are not added to the summary
        wandb.define_metric("Validation Epoch Loss", summary="none")
        wandb.define_metric("Validation Cross Entropy Epoch Loss", summary="none")
        wandb.define_metric("Validation Water Consistency Epoch Loss", summary="none")
        wandb.define_metric("Combined score", summary="none")
        wandb.define_metric("SIC r2_metric", summary="none")
        wandb.define_metric("SOD f1_metric", summary="none")
        wandb.define_metric("FLOE f1_metric", summary="none")
        wandb.define_metric("Water Consistency Accuarcy", summary="none")
        wandb.define_metric("Learning Rate", summary="none")

        wandb.save(str(args.config))
        print(colour_str('Save Config File', 'green'))

        start_time = time.time()

        print('-----------------------------------')
        print('Staring Validation')
        print('-----------------------------------')

        # this is for valset 1 visualization along with gt
        test('val', net, checkpoint_path, device, cfg.deepcopy(), train_options['validate_list'], 'Cross_Validation')

        print('-----------------------------------')
        print('Completed validation')
        print('-----------------------------------')

        valdiation_time = time.time()
        validation_elapsed_time = valdiation_time - start_time
        print(f'The code took {validation_elapsed_time} seconds to complete validation')

        print('-----------------------------------')
        print('Starting testing')
        print('-----------------------------------')

        # this is for test path along with gt after the gt has been released
        test('test', net, checkpoint_path, device, cfg.deepcopy(), train_options['test_list'], 'Test')

        print('-----------------------------------')
        print('Completed testing')
        print('-----------------------------------')

        test_time = time.time()
        test_elapsed_time = test_time - valdiation_time
        print(f'The code took {test_elapsed_time} seconds to complete testing')

        # finish the wandb run
        wandb.finish()

if __name__ == '__main__':
    main()
