import os
import torch
import numpy as np
import xarray as xr
import argparse
import json
import torch.nn.functional as F
from tqdm import tqdm
import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Python script to downsample polygon maps from raw data and embed polygon maps \
                                     and eggcodes to ready-to-train dataset')
    parser.add_argument('dataset_path', help='Dataset json path')
    parser.add_argument('dataset_save_path', help='Dataset json path')
    # parser.add_argument('prep_dataset_path', help='path to ready to train')
    # parser.add_argument('raw_dataset_path', help='path to raw dataset')
    # parser.add_argument('save_dataset_path', help='path to save dataset')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    directory_path = args.dataset_path + "*.nc"  # Update with the actual directory path and pattern

    nc_files = glob.glob(directory_path)

    # print("NC Files:", nc_files)

    list1 = []  # List for files without "_reference"
    list2 = []  # List for files with "_reference"

    for file_name in nc_files:
        if "_reference" in file_name:
            list2.append(file_name)
        else:
            list1.append(file_name)

    # print("List 1:", list1)
    # print("List 2:", list2)

    for file in tqdm(list1):
        sar_scene = xr.open_dataset(file, engine='h5netcdf', )  # Open the sar variable  scenes
        reference_file = file.replace(".nc", "_reference.nc")
        gt_scene = xr.open_dataset(reference_file, engine='h5netcdf')  # Open the sar variable  scenes
        sic = gt_scene['SIC']
        sod = gt_scene['SOD']
        floe = gt_scene['FLOE']
        # scene = scene.assign({'SIC': xr.DataArray(sic, dims=scene['polygon_icechart'].dims)})
        sar_scene_save = sar_scene.assign({'SIC': sic, 'SOD': sod, 'FLOE': floe})
        # sar_scene_save = sar_scene_save.assign({'SOD': sod})
        # sar_scene_save = sar_scene_save.assign({'FLOE': floe})
        # sar_scene.close()
        sar_scene_save.to_netcdf(os.path.join(args.dataset_save_path, os.path.basename(file)),
                                 mode='a', format='netcdf4', engine='h5netcdf')
        # Close the file
        sar_scene.close()
        gt_scene.close()
        sar_scene_save.close()
    #                              # f'{osp.splitext(osp.basename(cfg))[0]}
    #                              mode='w', format='netcdf4', engine='h5netcdf')

    # with open(args.dataset_json_path) as file:
    #     # datalist = json.loads(file.read())

    #     # Convert the original scene names to the preprocessed names.
    # datalist_prep = [file[17:32] + '_' + file[77:80] +
    #                                '_prep.nc' for file in datalist]
    # skipped_files_prep = []
    # skipped_files_raw = []

    # for raw_file, prep_file in tqdm(zip(datalist, datalist_prep)):

    #     try:
    #         raw_scene = xr.open_dataset(os.path.join(args.raw_dataset_path, raw_file))  # Open the raw scene
    #         prep_scene = xr.open_dataset(os.path.join(args.prep_dataset_path, prep_file))  # Open the prep scene
    #         polygon_chart = raw_scene['polygon_icechart'].values
    #         polygon_chart = np.nan_to_num(polygon_chart, False, nan=np.nan)
    #         polygon_chart = torch.from_numpy(polygon_chart)
    #         polygon_chart = F.max_pool2d(polygon_chart.unsqueeze(0).unsqueeze(0),
    #                                      kernel_size=2, stride=2).squeeze().squeeze()
    #         polygon_codes = raw_scene['polygon_codes']
    #         prep_scene = prep_scene.assign({'polygon_icechart': xr.DataArray(
    #             polygon_chart)})
    #         prep_scene = prep_scene.assign({'polygon_codes': xr.DataArray(
    #             polygon_codes, dims=polygon_codes.dims)})
    #         prep_scene.to_netcdf(os.path.join(args.save_dataset_path, prep_file),
    #                              # f'{osp.splitext(osp.basename(cfg))[0]}
    #                              mode='w', format='netcdf4', engine='h5netcdf')
    #         raw_scene.close()
    #         prep_scene.close()
    #     except:
    #         print(f"file skipped: {raw_file}")
    #         skipped_files_raw.append(raw_file)
    #         skipped_files_prep.append(prep_file)

    # with open("skipped_files.txt", "a") as file:
    #     for item1, item2 in skipped_files_prep, skipped_files_raw:
    #         file.write(str(item1) + "\n")
    #         file.write(str(item2) + "\n")

    # #         # Add the new charts to scene and add descriptions:
    # # scene = scene.assign({'SIC': xr.DataArray(sic, dims=scene['polygon_icechart'].dims)})
    # # scene = scene.assign({'SOD': xr.DataArray(sod, dims=scene['polygon_icechart'].dims)})
    # # scene = scene.assign({'FLOE': xr.DataArray(floe, dims=scene['polygon_icechart'].dims)})
if __name__ == '__main__':
    main()
