#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helping functions for 'introduction' and 'quickstart' notebooks."""

# -- File info -- #
__author__ = 'Andreas R. Stokholm'
__contributors__ = ''
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk']
__version__ = '1.0.0'
__date__ = '2022-10-17'

# -- Built-in modules -- #
import os
import json
# -- Third-party modules -- #
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
# from sklearn.metrics import r2_score, f1_score
from torchmetrics.functional import r2_score, f1_score
import segmentation_models_pytorch as smp
from tqdm import tqdm  # Progress bar
# -- Proprietary modules -- #

from utils import ICE_STRINGS, GROUP_NAMES
from unet import UNet, Sep_feat_dif_stages  # Convolutional Neural Network model
from swin_transformer import SwinTransformer  # Swin Transformer


def chart_cbar(ax, n_classes, chart, cmap='vridis'):
    """
    Create discrete colourbar for plot with the sea ice parameter class names.

    Parameters
    ----------
    n_classes: int
        Number of classes for the chart parameter.
    chart: str
        The relevant chart.
    """
    arranged = np.arange(0, n_classes)
    cmap = plt.get_cmap(cmap, n_classes - 1)
    # Get colour boundaries. -0.5 to center ticks for each color.
    norm = mpl.colors.BoundaryNorm(arranged - 0.5, cmap.N)
    arranged = arranged[:-1]  # Discount the mask class.
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=arranged, fraction=0.0485, pad=0.049, ax=ax)
    cbar.set_label(label=ICE_STRINGS[chart])
    cbar.set_ticklabels(list(GROUP_NAMES[chart].values()))


def compute_metrics(true, pred, charts, metrics, num_classes):
    """
    Calculates metrics for each chart and the combined score. true and pred must be 1d arrays of equal length.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels. Must be numpy array.
    pred :
        ndarray, 1d contains all predicted pixels. Must be numpy array.
    charts : List
        List of charts.
    metrics : Dict
        Stores metric calculation function and weight for each chart.

    Returns
    -------
    combined_score: float
        Combined weighted average score.
    scores: list
        List of scores for each chart.
    """
    scores = {}
    for chart in charts:
        if true[chart].ndim == 1 and pred[chart].ndim == 1:
            scores[chart] = torch.round(metrics[chart]['func'](
                true=true[chart], pred=pred[chart], num_classes=num_classes[chart]) * 100, decimals=3)

        else:
            print(f"true and pred must be 1D numpy array, got {true['SIC'].ndim} \
                and {pred['SIC'].ndim} dimensions with shape {true['SIC'].shape} and {pred.shape}, respectively")

    combined_score = compute_combined_score(scores=scores, charts=charts, metrics=metrics)

    return combined_score, scores


def r2_metric(true, pred, num_classes=None):
    """
    Calculate the r2 metric.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels. Must by numpy array.
    pred :
        ndarray, 1d contains all predicted pixels. Must by numpy array.
    num_classes :
        Num of classes in the dataset, this value is not used in this function but used in f1_metric function
        which requires num_classes argument. The reason it was included here was to keep the same structure.  


    Returns
    -------
    r2 : float
        The calculated r2 score.

    """
    r2 = r2_score(preds=pred, target=true)

    return r2


def f1_metric(true, pred, num_classes):
    """
    Calculate the weighted f1 metric.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels.
    pred :
        ndarray, 1d contains all predicted pixels.

    Returns
    -------
    f1 : float
        The calculated f1 score.

    """
    f1 = f1_score(target=true, preds=pred, average='weighted', task='multiclass', num_classes=num_classes)

    return f1


def water_edge_metric(outputs, options):

    # Convert ouput into water and not water
    for chart in options['charts']:

        outputs[chart] = torch.where(outputs[chart] > 0.0, 1.0, 0.0)

    # subtract them and absolute
    # perform mean
    water_edge_accuracy = 1 - torch.mean(torch.abs(outputs[options['charts'][0]]-outputs[options['charts'][1]])
                                         + torch.abs(outputs[options['charts'][1]]-outputs[options['charts'][2]])
                                         + torch.abs(outputs[options['charts'][2]]-outputs[options['charts'][0]]))

    return water_edge_accuracy


def water_edge_plot_overlay(output, mask, options):
    # Convert ouput into water and not water
    charts = options['charts']
    water_chart = {}
    for chart in charts:
        water_chart[chart] = np.where(output[chart] > 0.0, 0.75, 0.0)
        water_chart[chart][mask] = np.nan
        water_chart[chart] = water_chart[chart][..., np.newaxis]

    img = np.concatenate((water_chart[charts[0]], water_chart[charts[1]], water_chart[charts[2]]), axis=2,)

    return img


def compute_combined_score(scores, charts, metrics):
    """
    Calculate the combined weighted score.

    Parameters
    ----------
    scores : List
        Score for each chart.
    charts : List
        List of charts.
    metrics : Dict
        Stores metric calculation function and weight for each chart.

    Returns
    -------
    : float
        The combined weighted score.

    """
    combined_metric = 0
    sum_weight = 0
    for chart in charts:
        combined_metric += scores[chart] * metrics[chart]['weight']
        sum_weight += metrics[chart]['weight']

    return torch.round(combined_metric / sum_weight, decimals=3)


# -- functions to save models -- #
def save_best_model(cfg, train_options: dict, net, optimizer, scheduler, epoch: int):
    '''
    Saves the input model in the inside the directory "/work_dirs/"experiment_name"/
    The models with be save as best_model.pth.
    The following are stored inside best_model.pth
        model_state_dict
        optimizer_state_dict
        epoch
        train_options


    Parameters
    ----------
    cfg : mmcv.Config
        The config file object of mmcv
    train_options : Dict
        The dictory which stores the train_options from quickstart
    net :
        The pytorch model
    optimizer :
        The optimizer that the model uses.
    epoch: int
        The epoch number

    '''
    print('saving model....')
    config_file_name = os.path.basename(cfg.work_dir)
    # print(config_file_name)
    torch.save(obj={'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'train_options': train_options
                    },
               f=os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth'))
    print(f"model saved successfully at {os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth')}")

    return os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth')


def load_model(net, checkpoint_path, optimizer=None, scheduler=None):
    """
    Loads a PyTorch model from a checkpoint file and returns the model, optimizer, and scheduler.
    :param model: PyTorch model to load
    :param checkpoint_path: Path to the checkpoint file
    :param optimizer: PyTorch optimizer to load (optional)
    :param scheduler: PyTorch scheduler to load (optional)
    :return: If optimizer and scheduler are provided, return the model, optimizer, and scheduler.
    """

    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']

    return epoch


def rand_bbox(size, lam):
    '''
    Given the 4D dimensions of a batch (size), and the ratio 
    of the spatial dimension (lam) to be cut, returns a bounding box coordinates
    used for cutmix

    Parameters
    ----------
    size : 4D shape of the batch (N, C, H, W)
    lam : Ratio (portion) of the input to be cutmix'd

    Returns 
    ----------
    Bounding box (x1, y1, x2, y2)
    '''
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    # uniform
    cx = np.random.randint(H)
    cy = np.random.randint(W)

    bbx1 = np.clip(cx - cut_h // 2, 0, H)
    bby1 = np.clip(cy - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_w // 2, 0, W)

    return bbx1, bby1, bbx2, bby2


def slide_inference(img, net, options, mode):
    """
    Inference by sliding-window with overlap.


    Parameters
    ----------
    img : 4D shape of the batch (N, C', H, W)
    net : PyTorch model of nn.Module 
    options: configuration dictionary
    mode: either 'val' or 'test'

    Returns 
    ----------
    pred: Dictionary with SIC, SOD, and FLOE predictions of the batch  (N, C", H, W)
    """
    if mode == 'val':
        h_stride, w_stride = options['swin_hp']['val_stride']
    elif mode == 'test':
        h_stride, w_stride = options['swin_hp']['test_stride']
    else:
        raise 'Unrecognized mode'

    h_crop = options['patch_size']
    w_crop = options['patch_size']

    batch_size, _, h_img, w_img = img.size()
    SIC_channels = options['n_classes']['SIC']
    SOD_channels = options['n_classes']['SOD']
    FLOE_channels = options['n_classes']['FLOE']
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds_SIC = img.new_zeros((batch_size, SIC_channels, h_img, w_img))
    preds_SOD = img.new_zeros((batch_size, SOD_channels, h_img, w_img))
    preds_FLOE = img.new_zeros((batch_size, FLOE_channels, h_img, w_img))
    count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            crop_img_size = crop_img.size()
            if crop_img_size[2] < options['patch_size']:
                crop_height_pad = options['patch_size'] - crop_img_size[2]
            else:
                crop_height_pad = 0

            if crop_img_size[3] < options['patch_size']:
                crop_width_pad = options['patch_size'] - crop_img_size[3]
            else:
                crop_width_pad = 0

            if crop_height_pad > 0 or crop_width_pad > 0:
                crop_img = torch.nn.functional.pad(
                    crop_img, (0, crop_width_pad, 0, crop_height_pad), mode='constant', value=0)

            crop_seg_logit = net(crop_img)

            if crop_height_pad > 0:
                crop_seg_logit['SIC'] = crop_seg_logit['SIC'][:, :, :-crop_height_pad, :]
                crop_seg_logit['SOD'] = crop_seg_logit['SOD'][:, :, :-crop_height_pad, :]
                crop_seg_logit['FLOE'] = crop_seg_logit['FLOE'][:, :, :-crop_height_pad, :]
            if crop_width_pad > 0:
                crop_seg_logit['SIC'] = crop_seg_logit['SIC'][:, :, :, :-crop_width_pad]
                crop_seg_logit['SOD'] = crop_seg_logit['SOD'][:, :, :, :-crop_width_pad]
                crop_seg_logit['FLOE'] = crop_seg_logit['FLOE'][:, :, :, :-crop_width_pad]

            preds_SIC += torch.nn.functional.pad(crop_seg_logit['SIC'],
                                                 (int(x1), int(preds_SIC.shape[3] - x2), int(y1),
                                                  int(preds_SIC.shape[2] - y2)))
            preds_SOD += torch.nn.functional.pad(crop_seg_logit['SOD'],
                                                 (int(x1), int(preds_SOD.shape[3] - x2), int(y1),
                                                  int(preds_SOD.shape[2] - y2)))
            preds_FLOE += torch.nn.functional.pad(crop_seg_logit['FLOE'],
                                                  (int(x1), int(preds_FLOE.shape[3] - x2), int(y1),
                                                   int(preds_FLOE.shape[2] - y2)))

            count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0

    preds_SIC = preds_SIC / count_mat
    preds_SOD = preds_SOD / count_mat
    preds_FLOE = preds_FLOE / count_mat

    return {'SIC': preds_SIC,
            'SOD': preds_SOD,
            'FLOE': preds_FLOE}


class Slide_patches_index(data.Dataset):
    def __init__(self, h_img, w_img, h_crop, w_crop, h_stride, w_stride):
        super(Slide_patches_index, self).__init__()

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        self.patches_list = []

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                self.patches_list.append((y1, y2, x1, x2))

    def __getitem__(self, index):
        return self.patches_list[index]

    def __len__(self):
        return len(self.patches_list)


class Take_crops(data.Dataset):
    def __init__(self, img, patches):
        super(Take_crops, self).__init__()

        self.img = img
        self.patches = patches

    def __getitem__(self, index):
        y1, y2, x1, x2 = self.patches[index]

        return self.img[:, y1:y2, x1:x2]

    def __len__(self):
        return len(self.patches)


def batched_slide_inference(img, net, options, mode):
    """
    Inference by sliding-window with overlap.

    Parameters
    ----------
    img : 4D shape of the batch (N, C', H, W)
    net : PyTorch model of nn.Module 
    y_type: str, One of 'SIC', 'SOD', or 'FLOE'
    options: configuration dictionary

    Returns 
    ----------
    pred: Dictionary with SIC, SOD, and FLOE predictions of the batch  (N, C", H, W)
    """
    if mode == 'val':
        h_stride, w_stride = options['swin_hp']['val_stride']
    elif mode == 'test':
        h_stride, w_stride = options['swin_hp']['test_stride']
    else:
        raise 'Unrecognized mode'

    h_crop = options['patch_size']
    w_crop = options['patch_size']

    # ------------ Add Padding to the image to match with the patch size / stride
    _, _, h_img, w_img = img.size()
    height_pad = h_crop - h_img if h_img - h_crop < 0 else \
        (h_stride - (h_img - h_crop) % h_stride) % h_stride
    width_pad = w_crop - w_img if w_img - w_crop < 0 else \
        (w_stride - (w_img - w_crop) % w_stride) % w_stride
    if height_pad > 0 or width_pad > 0:
        img = torch.nn.functional.pad(
            img, (0, width_pad, 0, height_pad), mode='constant', value=0)

    # ------------ create dataloader and index track
    _, _, h_img, w_img = img.size()
    indexes = Slide_patches_index(h_img, w_img, h_crop, w_crop, h_stride, w_stride)
    samples = Take_crops(img.detach().cpu().numpy()[0], indexes.patches_list)
    samples_dataloader = data.DataLoader(dataset=samples, batch_size=options['batch_size']*4,
                                         shuffle=False, num_workers=options['num_workers_val'])

    n_batches = len(samples_dataloader)
    data_iterator = iter(samples_dataloader)
    idx_iterator = iter(indexes)

    SIC_channels = options['n_classes']['SIC']
    SOD_channels = options['n_classes']['SOD']
    FLOE_channels = options['n_classes']['FLOE']
    preds_SIC = img.new_zeros((SIC_channels, h_img, w_img))
    preds_SOD = img.new_zeros((SOD_channels, h_img, w_img))
    preds_FLOE = img.new_zeros((FLOE_channels, h_img, w_img))
    count_mat = img.new_zeros((h_img, w_img))

    for i in range(n_batches):

        # ------------ Take data
        crop_imgs = next(data_iterator)
        crop_imgs = crop_imgs.to(img.device)

        # ------------ Forward
        crop_seg_logit = net(crop_imgs)

        # ------------ LOCATE PREDICTED LOGITS ON THE WHOLE SCENE
        for j in range(crop_imgs.shape[0]):
            y1, y2, x1, x2 = next(idx_iterator)

            preds_SIC[:, y1:y2, x1:x2] += crop_seg_logit['SIC'][j, :, 0:(y2-y1), 0:(x2-x1)]
            preds_SOD[:, y1:y2, x1:x2] += crop_seg_logit['SOD'][j, :, 0:(y2-y1), 0:(x2-x1)]
            preds_FLOE[:, y1:y2, x1:x2] += crop_seg_logit['FLOE'][j, :, 0:(y2-y1), 0:(x2-x1)]

            count_mat[y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0

    preds_SIC = preds_SIC / count_mat
    preds_SOD = preds_SOD / count_mat
    preds_FLOE = preds_FLOE / count_mat

    # ------------ Remove pad
    preds_SIC = preds_SIC[:, :-height_pad, :-width_pad].unsqueeze(0)
    preds_SOD = preds_SOD[:, :-height_pad, :-width_pad].unsqueeze(0)
    preds_FLOE = preds_FLOE[:, :-height_pad, :-width_pad].unsqueeze(0)

    return {'SIC': preds_SIC,
            'SOD': preds_SOD,
            'FLOE': preds_FLOE}


def class_decider(output, train_options, chart):

    # normal
    if (train_options['binary_water_classifier'] == False):
        if output.size(3) == 1:
            output = torch.round(output.squeeze())
            output = torch.clamp(output, min=0, max=train_options
                                 ['n_classes'][chart])
        else:
            output = torch.argmax(output, dim=1).squeeze()
        return output

    # if regression head    return output
    # class water
    else:
        probability = torch.nn.Softmax(dim=1)(output)
        water = probability[:, 0, :, :]
        not_water = torch.sum(probability, dim=1) - water
        class_output = water <= not_water
        without_water = probability[:, 1:, :, :]
        class_output_without_water = torch.argmax(without_water, dim=1) + 1
        class_output = class_output_without_water * class_output

        return class_output.squeeze()


def compute_classwise_f1score(true, pred, charts, num_classes):
    """ This function computes the classwise evaluation score for each task and stores them in a dic

    Args:
        true (dictionary): The true tensor as value and chart tensor as key
        pred (dictionary): The pred tensor as value and chart tensor as key
        charts (list): list of charts
        num_classes (dictionary): key = chart , value = num_class

    Returns:
        dictionary: returns score_dictionary
    """
    score = {}
    for chart in charts:
        score[chart] = f1_score(target=true[chart], preds=pred[chart], average='none',
                                task='multiclass', num_classes=num_classes[chart])
    return score


def create_train_validation_and_test_scene_list(train_options):
    '''
    Creates the a train and validation scene list. Adds these two list to the config file train_options

    '''

    # Train ------------
    with open(train_options['path_to_env'] + train_options['train_list_path']) as file:
        train_options['train_list'] = json.loads(file.read())

    # Convert the original scene names to the preprocessed names.
    train_options['train_list'] = [file[17:32] + '_' + file[77:80] +
                                   '_prep.nc' for file in train_options['train_list']]

    # Validation ---------
    if train_options['cross_val_run']:
        # Select a random number of validation scenes with the same seed. Feel free to change the seed.et
        train_options['validate_list'] = np.random.choice(np.array(
            train_options['train_list']), size=train_options['p-out'], replace=False)
    else:
        # load validation list
        with open(train_options['path_to_env'] + train_options['val_path']) as file:
            train_options['validate_list'] = json.loads(file.read())
        # Convert the original scene names to the preprocessed names.
        train_options['validate_list'] = [file[17:32] + '_' + file[77:80] +
                                          '_prep.nc' for file in train_options['validate_list']]

    # Remove the validation scenes from the train list.
    train_options['train_list'] = [scene for scene in train_options['train_list']
                                   if scene not in train_options['validate_list']]
    
    # Test ----------
    with open(train_options['path_to_env'] + train_options['test_path']) as file:
        train_options['test_list'] = json.loads(file.read())
        train_options['test_list'] = [file[17:32] + '_' + file[77:80] + '_prep.nc'
                                      for file in train_options['test_list']]
    print('Options initialised')


def get_scheduler(train_options, optimizer):
    if train_options['scheduler']['type'] == 'CosineAnnealingLR':
        T_max = train_options['epochs']*train_options['epoch_len']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max,
                                                               eta_min=train_options['scheduler']['lr_min'])
    elif train_options['scheduler']['type'] == 'CosineAnnealingWarmRestartsLR':
        # T_max = train_options['epochs']*train_options['epoch_len']
        T_0 = train_options['scheduler']['EpochsPerRestart']*train_options['epoch_len']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0,
                                                                         T_mult=train_options['scheduler']['RestartMult'],
                                                                         eta_min=train_options['scheduler']['lr_min'],
                                                                         last_epoch=-1,
                                                                         verbose=False)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=5, last_epoch=- 1,
                                                        verbose=False)
    return scheduler


def get_optimizer(train_options, net):
    if train_options['optimizer']['type'] == 'Adam':
        optimizer = torch.optim.Adam(list(net.parameters()), lr=train_options['optimizer']['lr'],
                                     betas=(train_options['optimizer']['b1'], train_options['optimizer']['b2']),
                                     weight_decay=train_options['optimizer']['weight_decay'])

    elif train_options['optimizer']['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(list(net.parameters()), lr=train_options['optimizer']['lr'],
                                      betas=(train_options['optimizer']['b1'], train_options['optimizer']['b2']),
                                      weight_decay=train_options['optimizer']['weight_decay'])
    else:
        optimizer = torch.optim.SGD(list(net.parameters()), lr=train_options['optimizer']['lr'],
                                    momentum=train_options['optimizer']['momentum'],
                                    dampening=train_options['optimizer']['dampening'],
                                    weight_decay=train_options['optimizer']['weight_decay'],
                                    nesterov=train_options['optimizer']['nesterov'])
    return optimizer


def get_loss(loss, chart=None, **kwargs):
    # TODO Fix Dice loss, Jacard loss,  MCC loss, SoftBCEWithLogitsLoss,
    """_summary_

    Args:
        loss (str): the name of the loss
    Returns:
        loss: The corresponding
    """
    if loss == 'DiceLoss':
        kwargs.pop('type')
        loss = smp.losses.DiceLoss(**kwargs)
    elif loss == 'FocalLoss':
        kwargs.pop('type')
        loss = smp.losses.FocalLoss(**kwargs)
    elif loss == 'JaccardLoss':
        raise NotImplementedError
        kwargs.pop('type')
        loss = smp.losses.JaccardLoss(**kwargs)
    elif loss == 'LovaszLoss':
        kwargs.pop('type')
        loss = smp.losses.LovaszLoss(**kwargs)
    elif loss == 'MCCLoss':
        kwargs.pop('type')
        loss = smp.losses.MCCLoss(**kwargs)
    elif loss == 'SoftBCEWithLogitsLoss':
        raise NotImplementedError
        kwargs.pop('type')
        loss = smp.losses.SoftBCEWithLogitsLoss(**kwargs)
    elif loss == 'SoftCrossEntropyLoss':
        raise NotImplementedError
        kwargs.pop('type')
        loss = smp.losses.SoftCrossEntropyLoss(**kwargs)
    elif loss == 'TverskyLoss':
        kwargs.pop('type')
        loss = smp.losses.TverskyLoss(**kwargs)
    elif loss == 'CrossEntropyLoss':
        kwargs.pop('type')
        loss = torch.nn.CrossEntropyLoss(**kwargs)
    elif loss == 'BinaryCrossEntropyLoss':
        raise NotImplementedError
        kwargs.pop('type')
        loss = torch.nn.BCELoss(**kwargs)
    elif loss == 'OrderedCrossEntropyLoss':
        from losses import OrderedCrossEntropyLoss
        kwargs.pop('type')
        loss = OrderedCrossEntropyLoss(**kwargs)
    elif loss == 'MSELossFromLogits':
        from losses import MSELossFromLogits
        kwargs.pop('type')
        loss = MSELossFromLogits(chart=chart, **kwargs)
    elif loss == 'MSELoss':
        kwargs.pop('type')
        loss = torch.nn.MSELoss(**kwargs)
    elif loss == 'MSELossWithIgnoreIndex':
        from losses import MSELossWithIgnoreIndex
        kwargs.pop('type')
        loss = MSELossWithIgnoreIndex(**kwargs)
    else:
        raise ValueError(f'The given loss \'{loss}\' is unrecognized or Not implemented')

    return loss


def get_model(train_options, device):
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
    return net
