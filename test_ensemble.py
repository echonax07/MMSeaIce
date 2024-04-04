
# # AutoICE - test model and prepare upload package
# This notebook tests the 'best_model', created in the quickstart notebook,
# with the tests scenes exempt of reference data.
# The model outputs are stored per scene and chart in an xarray Dataset in individual Dataarrays.
# The xarray Dataset is saved and compressed in an .nc file ready to be uploaded to the AI4EO.eu platform.
# Finally, the scene chart inference is shown.
#
# The first cell imports necessary packages:

# -- Built-in modules -- #
import argparse
import json
import os
import os.path as osp
import pathlib
import glob

# -- Third-part modules -- #
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm
from icecream import ic
from mmcv import Config, mkdir_or_exist

# --Proprietary modules -- #
from functions import chart_cbar
from loaders import AI4ArcticChallengeTestDataset, get_variable_options
from functions import slide_inference, batched_slide_inference
import wandb
from unet import UNet  # Convolutional Neural Network model
from swin_transformer import SwinTransformer  # Swin Transformer
from quickstart import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble cross validation model by avg over logits')

    # Mandatory arguments
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--wandb-project', required=True, help='Name of wandb project')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    ic(args.config)
    cfg = Config.fromfile(args.config)
    train_options = cfg.train_options
    assert train_options['cross_val_run'] == True, 'The given config file has \'cross_val_run\' set to False, it should be True'
    # Get options for variables, amsrenv grid, cropping and upsampling.
    train_options = get_variable_options(train_options)
    # cfg['experiment_name']=
    # cfg.env_dict = {}
    # work_dir is determined in this priority: CLI > segment in file > filename

    run_name = 'ensemble'
    cfg.work_dir = osp.join('./work_dir',
                            osp.splitext(osp.basename(args.config))[0], run_name)

    mkdir_or_exist(osp.abspath(cfg.work_dir))

    checkpoints = glob.glob(osp.join('./work_dir',
                                     osp.splitext(osp.basename(args.config))[0] + '/*/*.pth'), recursive=True)

    experiment_name = run_name
    artifact = wandb.Artifact(experiment_name, 'dataset')
    table = wandb.Table(columns=['ID', 'Image'])

    train_options = cfg.train_options
    train_options = get_variable_options(train_options)

    device = torch.device(f"cuda:{train_options['gpu_id']}")

    net = get_model(train_options, device)

    # Initialize dataloader and dataset
    with open(train_options['path_to_env'] + 'datalists/testset.json') as file:
        train_options['test_list'] = json.loads(file.read())
        train_options['test_list'] = [file[17:32] + '_' + file[77:80] + '_prep.nc'
                                      for file in train_options['test_list']]
        # The test data is stored in a separate folder inside the training data.
        upload_package = xr.Dataset()  # To store model outputs.
        dataset = AI4ArcticChallengeTestDataset(
            options=train_options, files=train_options['test_list'], mode='test')
        asid_loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)
        print('Setup ready')

    with wandb.init(name=run_name, group=osp.splitext(osp.basename(args.config))[0], project=args.wandb_project,
                    entity="ai4arctic", config=train_options):

        for inf_x, _, masks, scene_name, original_size in tqdm(iterable=asid_loader,
                                                               total=len(train_options['test_list']), colour='green', position=0):
            scene_name = scene_name[:19]  # Removes the _prep.nc from the name.
            torch.cuda.empty_cache()
            inf_x = inf_x.to(device, non_blocking=True)

            # initialize empty dic(k) with keys as charts and values as list
            model_logits = {}
            for chart in train_options['charts']:
                model_logits[chart] = []

            for i, checkpoint in enumerate(checkpoints):
                weights = torch.load(checkpoint)['model_state_dict']
                net.load_state_dict(weights)
                print(f'Model {i} successfully loaded.')
                net.eval()

                output = net(inf_x)
                # turn off requrie grad
                for chart in train_options['charts']:
                    output[chart] = output[chart].detach()
                masks_int = masks.to(torch.uint8)
                masks_int = torch.nn.functional.interpolate(masks_int.unsqueeze(
                    0).unsqueeze(0), size=original_size, mode='nearest').squeeze().squeeze()
                masks = torch.gt(masks_int, 0)

                # masks = torch.nn.functional.interpolate(masks.unsqueeze(0).unsqueeze(0), size = original_size, mode = 'nearest').squeeze().squeeze()
                # Upsample to match the correct size
                for chart in train_options['charts']:
                    if output[chart].size(3) == 1:  # regression output
                        output[chart] = output[chart].permute(0, 3, 1, 2)
                        model_logits[chart].append(output[chart].cpu())
                    else:    # normal output
                        if train_options['ensemble_after_softmax']:
                            model_logits[chart].append(torch.nn.functional.softmax(output[chart].cpu(), dim=1))
                        else:
                            model_logits[chart].append(output[chart].cpu())

                # override model logits which is a list to a tensor for each chart
            for chart in train_options['charts']:
                model_logits[chart] = torch.mean(torch.cat(model_logits[chart], dim=0), dim=0).unsqueeze(dim=0)

            for chart in train_options['charts']:
                if model_logits[chart].size(1) == 1:
                    model_logits[chart] = torch.round(model_logits[chart].float()).squeeze().cpu()
                    model_logits[chart] = torch.clamp(model_logits[chart], min=0,
                                                      max=train_options['n_classes'][chart])

                else:
                    if train_options['ensemble_after_softmax']:
                        model_logits[chart] = torch.argmax(model_logits[chart], dim=1).squeeze().cpu()
                    else:
                        model_logits[chart] = torch.argmax(torch.nn.functional.softmax(
                            model_logits[chart]), dim=1).squeeze().cpu().numpy()

                if train_options['down_sample_scale'] != 1:
                    model_logits[chart] = torch.nn.functional.interpolate(
                        model_logits[chart].unsqueeze(dim=0).unsqueeze(dim=0).to(torch.float32), size=original_size, mode='nearest').squeeze().squeeze().numpy()

                upload_package[f"{scene_name}_{chart}"] = xr.DataArray(name=f"{scene_name}_{chart}", data=model_logits[chart].astype('uint8'),
                                                                       dims=(f"{scene_name}_{chart}_dim0", f"{scene_name}_{chart}_dim1"))

            # - Show the scene inference.
            fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20, 20))
            for idx, chart in enumerate(train_options['charts']):
                for j in range(0, 2):
                    ax = axs[j]
                    img = torch.squeeze(inf_x, dim=0).cpu().numpy()[j]
                    if j == 0:
                        ax.set_title('HH')
                    else:
                        ax.set_title('HV')
                    ax.imshow(img, cmap='gray')
                ax = axs[idx+2]
                model_logits[chart] = model_logits[chart].astype(float)
                model_logits[chart][masks] = np.nan
                ax.imshow(model_logits[chart], vmin=0, vmax=train_options['n_classes']
                          [chart] - 2, cmap='jet', interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                chart_cbar(ax=ax, n_classes=train_options['n_classes'][chart], chart=chart, cmap='jet')

            plt.suptitle(f"Scene: {scene_name}", y=0.65)
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.5, hspace=-0)
            fig.savefig(f"{osp.join(cfg.work_dir,scene_name)}_ensemble.png",
                        format='png', dpi=128, bbox_inches="tight")
            plt.close('all')
            table.add_data(scene_name, wandb.Image(f"{osp.join(cfg.work_dir,scene_name)}_ensemble.png"))

        artifact.add(table, experiment_name+'_test')
        wandb.log_artifact(artifact)
        # - Save upload_package with zlib compression.
        print('Saving upload_package. Compressing data with zlib.')
        compression = dict(zlib=True, complevel=1)
        encoding = {var: compression for var in upload_package.data_vars}
        upload_package.to_netcdf(osp.join(cfg.work_dir, f'{experiment_name}_upload_package.nc'),
                                 # f'{osp.splitext(osp.basename(cfg))[0]}
                                 mode='w', format='netcdf4', engine='h5netcdf', encoding=encoding)
        print('Testing completed.')
        print("File saved succesfully at", osp.join(cfg.work_dir, f'{experiment_name}_upload_package.nc'))
        wandb.save(osp.join(cfg.work_dir, f'{experiment_name}_upload_package.nc'))
