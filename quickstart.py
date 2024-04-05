__author__ = 'Muhammed Patel'
__contributor__ = 'Xinwwei chen, Fernando Pena Cantu,Javier Turnes, Eddie Park'
__copyright__ = ['university of waterloo']
__contact__ = ['m32patel@uwaterloo.ca', 'xinweic@uwaterloo.ca']
__version__ = '1.0.0'
__date__ = '2024-04-05'

import argparse
import json
import random
import os
import os.path as osp
import shutil
from icecream import ic
import pathlib
import warnings

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
    parser.add_argument('--wandb-project', required=True, help='Name of wandb project')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--seed', default=None,
                        help='the seed to use, if not provided, seed from config file will be taken')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume-from', type=pathlib.Path, default=None,
                       help='Resume Training from checkpoint, it will use the \
                        optimizer and schduler defined on checkpoint')
    group.add_argument('--finetune-from', type=pathlib.Path, default=None,
                       help='Start new tranining using the weights from checkpoitn')

    args = parser.parse_args()

    return args


def train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer, scheduler, start_epoch=0):
    '''
    Trains the model.

    '''
    best_combined_score = -np.Inf  # Best weighted model score.

    loss_ce_functions = {chart: get_loss(train_options['chart_loss'][chart]['type'], chart=chart, **train_options['chart_loss'][chart])
                         for chart in train_options['charts']}

    loss_water_edge_consistency = WaterConsistencyLoss()
    print('Training...')
    # -- Training Loop -- #
    for epoch in tqdm(iterable=range(start_epoch, train_options['epochs'])):
        # gc.collect()  # Collect garbage to free memory.
        train_loss_sum = torch.tensor([0.])  # To sum the training batch losses during the epoch.
        cross_entropy_loss_sum = torch.tensor([0.])  # To sum the training cross entropy batch losses during the epoch.
        # To sum the training edge consistency batch losses during the epoch.
        edge_consistency_loss_sum = torch.tensor([0.])

        val_loss_sum = torch.tensor([0.])  # To sum the validation batch losses during the epoch.
        # To sum the validation cross entropy batch losses during the epoch.
        val_cross_entropy_loss_sum = torch.tensor([0.])
        # To sum the validation cedge consistency batch losses during the epoch.
        val_edge_consistency_loss_sum = torch.tensor([0.])
        net.train()  # Set network to evaluation mode.

        # Loops though batches in queue.
        for i, (batch_x, batch_y) in enumerate(tqdm(iterable=dataloader_train, total=train_options['epoch_len'],
                                                    colour='red')):
            # torch.cuda.empty_cache()  # Empties the GPU cache freeing up memory.
            train_loss_batch = torch.tensor([0.]).to(device)  # Reset from previous batch.
            edge_consistency_loss = torch.tensor([0.]).to(device)
            cross_entropy_loss = torch.tensor([0.]).to(device)
            # - Transfer to device.
            batch_x = batch_x.to(device, non_blocking=True)

            # - Mixed precision training. (Saving memory)
            with torch.cuda.amp.autocast():
                # - Forward pass.
                output = net(batch_x)
                # breakpoint()
                # - Calculate loss.
                for chart, weight in zip(train_options['charts'], train_options['task_weights']):

                    if train_options['edge_consistency_loss'] != 0:
                        edge_consistency_loss = loss_water_edge_consistency(output)

                    cross_entropy_loss += weight * loss_ce_functions[chart](
                        output[chart], batch_y[chart].to(device))

            if train_options['edge_consistency_loss'] != 0:
                a = train_options['edge_consistency_loss']
                edge_consistency_loss = a*loss_water_edge_consistency(output)
                train_loss_batch = cross_entropy_loss + edge_consistency_loss
            else:
                train_loss_batch = cross_entropy_loss

            # - Reset gradients from previous pass.
            optimizer.zero_grad()

            # - Backward pass.
            train_loss_batch.backward()

            # - Optimizer step
            optimizer.step()

            # - Scheduler step
            scheduler.step()

            # - Add batch loss.
            train_loss_sum += train_loss_batch.detach().item()
            cross_entropy_loss_sum += cross_entropy_loss.detach().item()
            edge_consistency_loss_sum += edge_consistency_loss.detach().item()

        # - Average loss for displaying
        train_loss_epoch = torch.true_divide(train_loss_sum, i + 1).detach().item()
        cross_entropy_epoch = torch.true_divide(cross_entropy_loss_sum, i + 1).detach().item()
        edge_consistency_epoch = torch.true_divide(edge_consistency_loss_sum, i + 1).detach().item()

        # del output, batch_x, batch_y # Free memory.
        # del loss_sum

        # -- Validation Loop -- #
        # For printing after the validation loop.

        # - Stores the output and the reference pixels to calculate the scores after inference on all the scenes.
        outputs_flat = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
        inf_ys_flat = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
        # Outputs mask by train fill values
        outputs_tfv_mask = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
        net.eval()  # Set network to evaluation mode.
        print('Validating...')
        # - Loops though scenes in queue.
        for i, (inf_x, inf_y, cfv_masks, tfv_mask, name, original_size) in enumerate(tqdm(iterable=dataloader_val,
                                                                            total=len(train_options['validate_list']),
                                                                            colour='green')):
            torch.cuda.empty_cache()
            # Reset from previous batch.
            # train fill value mask
            # tfv_mask = (inf_x.squeeze()[0, :, :] == train_options['train_fill_value']).squeeze()
            val_loss_batch = torch.tensor([0.]).to(device)
            val_edge_consistency_loss = torch.tensor([0.]).to(device)
            val_cross_entropy_loss = torch.tensor([0.]).to(device)
            # - Ensures that no gradients are calculated, which otherwise take up a lot of space on the GPU.
            with torch.no_grad(), torch.cuda.amp.autocast():
                inf_x = inf_x.to(device, non_blocking=True)
                if train_options['model_selection'] == 'swin':
                    output = slide_inference(inf_x, net, train_options, 'val')
                    # output = batched_slide_inference(inf_x, net, train_options, 'val')
                else:
                    output = net(inf_x)

                for chart, weight in zip(train_options['charts'], train_options['task_weights']):

                    val_cross_entropy_loss += weight * loss_ce_functions[chart](output[chart],
                                                                                inf_y[chart].unsqueeze(0).long().to(device))

                if train_options['edge_consistency_loss'] != 0:
                    a = train_options['edge_consistency_loss']
                    val_edge_consistency_loss = a*loss_water_edge_consistency(output)

            val_loss_batch = val_cross_entropy_loss + val_edge_consistency_loss

            # - Final output layer, and storing of non masked pixels.
            for chart in train_options['charts']:
                output[chart] = class_decider(output[chart], train_options, chart)
                # output[chart] = torch.argmax(
                #     output[chart], dim=1).squeeze()
                outputs_flat[chart] = torch.cat((outputs_flat[chart], output[chart][~cfv_masks[chart]]))
                outputs_tfv_mask[chart] = torch.cat((outputs_tfv_mask[chart], output[chart][~tfv_mask]))
                inf_ys_flat[chart] = torch.cat((inf_ys_flat[chart], inf_y[chart]
                                                [~cfv_masks[chart]].to(device, non_blocking=True)))
            # - Add batch loss.
            val_loss_sum += val_loss_batch.detach().item()
            val_cross_entropy_loss_sum += val_cross_entropy_loss.detach().item()
            val_edge_consistency_loss_sum += val_edge_consistency_loss.detach().item()

        # - Average loss for displaying
        val_loss_epoch = torch.true_divide(val_loss_sum, i + 1).detach().item()
        val_cross_entropy_epoch = torch.true_divide(val_cross_entropy_loss_sum, i + 1).detach().item()
        val_edge_consistency_epoch = torch.true_divide(val_edge_consistency_loss_sum, i + 1).detach().item()

        # - Compute the relevant scores.
        print('Computing Metrics on Val dataset')
        combined_score, scores = compute_metrics(true=inf_ys_flat, pred=outputs_flat, charts=train_options['charts'],
                                                 metrics=train_options['chart_metric'], num_classes=train_options['n_classes'])

        water_edge_accuarcy = water_edge_metric(outputs_tfv_mask, train_options)

        if train_options['compute_classwise_f1score']:
            from functions import compute_classwise_f1score
            # dictionary key = chart, value = tensor; e.g  key = SOD, value = tensor([0, 0.2, 0.4, 0.2, 0.1])
            classwise_scores = compute_classwise_f1score(true=inf_ys_flat, pred=outputs_flat,
                                                         charts=train_options['charts'], num_classes=train_options['n_classes'])
        print("")
        print(f"Epoch {epoch} score:")

        for chart in train_options['charts']:
            print(f"{chart} {train_options['chart_metric'][chart]['func'].__name__}: {scores[chart]}%")

            # Log in wandb the SIC r2_metric, SOD f1_metric and FLOE f1_metric
            wandb.log({f"{chart} {train_options['chart_metric'][chart]['func'].__name__}": scores[chart]}, step=epoch)

            # if classwise_f1score is True,
            if train_options['compute_classwise_f1score']:
                for index, class_score in enumerate(classwise_scores[chart]):
                    wandb.log({f"{chart}/Class: {index}": class_score.item()}, step=epoch)
                print(f"{chart} F1 score:", classwise_scores[chart])

        print(f"Combined score: {combined_score}%")
        print(f"Train Epoch Loss: {train_loss_epoch:.3f}")
        print(f"Train Cross Entropy Epoch Loss: {cross_entropy_epoch:.3f}")
        print(f"Train Water Consistency Epoch Loss: {edge_consistency_epoch:.3f}")
        print(f"Validation Epoch Loss: {val_loss_epoch:.3f}")
        print(f"Validation Cross Entropy Epoch Loss: {val_cross_entropy_epoch:.3f}")
        print(f"Validation val_edge_consistency_loss: {val_edge_consistency_epoch:.3f}")
        print(f"Water edge Accuarcy: {water_edge_accuarcy}")

        # Log combine score and epoch loss to wandb
        wandb.log({"Combined score": combined_score,
                   "Train Epoch Loss": train_loss_epoch,
                   "Train Cross Entropy Epoch Loss": cross_entropy_epoch,
                   "Train Water Consistency Epoch Loss": edge_consistency_epoch,
                   "Validation Epoch Loss": val_loss_epoch,
                   "Validation Cross Entropy Epoch Loss": val_cross_entropy_epoch,
                   "Validation Water Consistency Epoch Loss": val_edge_consistency_epoch,
                   "Water Consistency Accuarcy": water_edge_accuarcy,
                   "Learning Rate": optimizer.param_groups[0]["lr"]}, step=epoch)

        # If the scores is better than the previous epoch, then save the model and rename the image to best_validation.

        if combined_score > best_combined_score:
            best_combined_score = combined_score

            # Log the best combine score, and the metrics that make that best combine score in summary in wandb.
            wandb.run.summary[f"While training/Best Combined Score"] = best_combined_score
            wandb.run.summary[f"While training/Water Consistency Accuarcy"] = water_edge_accuarcy
            for chart in train_options['charts']:
                wandb.run.summary[f"While training/{chart} {train_options['chart_metric'][chart]['func'].__name__}"] = scores[chart]
            wandb.run.summary[f"While training/Train Epoch Loss"] = train_loss_epoch

            # Save the best model in work_dirs
            model_path = save_best_model(cfg, train_options, net, optimizer, scheduler, epoch)

            wandb.save(model_path)

    del inf_ys_flat, outputs_flat  # Free memory.
    return model_path


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


def main():
    args = parse_args()
    ic(args.config)
    cfg = Config.fromfile(args.config)
    train_options = cfg.train_options
    # Get options for variables, amsrenv grid, cropping and upsampling.
    train_options = get_variable_options(train_options)
    # generate wandb run id, to be used to link the run with test_upload
    id = wandb.util.generate_id()

    # Set the seed if not -1
    if train_options['seed'] != -1 and args.seed == None:
        # set seed for everything
        if args.seed != None:
            seed = int(args.seed)
        else:
            seed = train_options['seed']
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = True
        print(f"Seed: {seed}")
    else:
        print("Random Seed Chosen")
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
    # cfg_path = osp.join(cfg.work_dir, osp.basename(args.config))
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

    net = get_model(train_options, device)

    if train_options['compile_model']:
        net = torch.compile(net)

    optimizer = get_optimizer(train_options, net)

    scheduler = get_scheduler(train_options, optimizer)

    if args.resume_from is not None:
        print(f"\033[91m Resuming work from {args.resume_from}\033[0m")
        epoch_start = load_model(net, args.resume_from, optimizer, scheduler)
    elif args.finetune_from is not None:
        print(f"\033[91m Finetune model from {args.finetune_from}\033[0m")
        _ = load_model(net, args.finetune_from)

    # initialize wandb run

    if not train_options['cross_val_run']:
        wandb.init(name=osp.splitext(osp.basename(args.config))[0], project=args.wandb_project,
                   entity="ai4arctic", config=train_options, id=id, resume="allow")
    else:
        wandb.init(name=osp.splitext(osp.basename(args.config))[0]+'-'+run_name, group=osp.splitext(osp.basename(args.config))[0], project=args.wandb_project,
                   entity="ai4arctic", config=train_options, id=id, resume="allow")

    # Define the metrics and make them such that they are not added to the summary
    wandb.define_metric("Train Epoch Loss", summary="none")
    wandb.define_metric("Train Cross Entropy Epoch Loss", summary="none")
    wandb.define_metric("Train Water Consistency Epoch Loss", summary="none")
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

    create_train_validation_and_test_scene_list(train_options)

    dataloader_train, dataloader_val = create_dataloaders(train_options)

    # Update Config
    wandb.config['validate_list'] = train_options['validate_list']


    print('Data setup complete.')

    # ## Example of model training and validation loop
    # A simple model training loop following by a simple validation loop. Validation is carried out on full scenes,
    #  i.e. no cropping or stitching. If there is not enough space on the GPU, then try to do it on the cpu.
    #  This can be done by using 'net = net.cpu()'.


    print('-----------------------------------')
    print('Starting Training')
    print('-----------------------------------')
    if args.resume_from is not None:
        checkpoint_path = train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer,
                                scheduler, epoch_start)
    else:
        checkpoint_path = train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer,
                                scheduler)


    print('-----------------------------------')
    print('Training Complete')
    print('-----------------------------------')



    print('-----------------------------------')
    print('Staring Validation with best model')
    print('-----------------------------------')

    # this is for valset 1 visualization along with gt
    test('val', net, checkpoint_path, device, cfg.deepcopy(), train_options['validate_list'], 'Cross Validation')


    print('-----------------------------------')
    print('Completed validation')
    print('-----------------------------------')




    print('-----------------------------------')
    print('Starting testing with best model')
    print('-----------------------------------')

    # this is for test path along with gt after the gt has been released
    test('test', net, checkpoint_path, device, cfg.deepcopy(), train_options['test_list'], 'Test')

    print('-----------------------------------')
    print('Completed testing')
    print('-----------------------------------')


    # finish the wandb run
    wandb.finish()


if __name__ == '__main__':
    main()
