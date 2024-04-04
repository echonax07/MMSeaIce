import os
import xarray as xr
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

import cmocean.cm as cmo
from r2t_vis import percentile_clip, plot_chart_data

"""
Command-line script to write NetCDF data (imagery + ice charts) to file.

Image bands (HH, HV & IA) are written to individual .jpg files, with HH and HV
bands clipped to percentiles to improve contrast for visualization.

Plots of ice charts (SIC, SOD & FLOE) are written to a single .png image per scene.

Usage:   python export_data.py {in_dir} {out_dir}

"""

def intify_band(band, clip=False):
    """
    Scale image data to range 0-255, mapping 0-filled values (e.g. land) to 128.
    """

    scaled = np.zeros(band.shape)

    # First clip intensity distribution for better contrast (optional)
    if clip: band = percentile_clip(band, [2, 98])

    # Now scale to [0, 255]
    fillidcs = band == 0
    negidcs = band < 0
    posidcs = band > 0

    b_min = np.nanmin(band)
    b_max = np.nanmax(band)

    # Map negative values to 0-127;
    scaled[negidcs] = 127 * (band[negidcs] - b_min) / (0 - b_min)

    # Map 0 values to 128;
    scaled[fillidcs] = 128

    # Map positive values to 129-255.
    scaled[posidcs] = (126 * (band[posidcs] - 0) / (b_max - 0)) + 129

    return np.uint8(scaled)


def main():
    parser = argparse.ArgumentParser(
        prog='extract_imagery',
        description='Extract image channels in .nc files and export as high-contrast .png'
    )
    parser.add_argument('input_dir', help='folder containing input .nc files')
    parser.add_argument('output_dir', help='folder to write .png files to')

    args = parser.parse_args()
    
    in_dir = args.input_dir.replace('\\', '/')
    if not in_dir[-1] == '/': in_dir += '/'
    out_dir = args.output_dir.replace('\\', '/')
    if not out_dir[-1] == '/': out_dir += '/'

    if not os.path.exists(out_dir): os.mkdir(out_dir)

    print(f"Extracting imagery from {in_dir} to {out_dir}...")
    for fn in tqdm(os.listdir(in_dir)):
        if fn.split('.')[-1] != 'nc': continue
        fp = in_dir + fn

        with xr.open_dataset(fp) as data:
            HH = data.nersc_sar_primary.values
            HV = data.nersc_sar_secondary.values
            IA = data.sar_incidenceangle.values

            SIC = data.SIC
            SOD = data.SOD
            FLOE = data.FLOE
            sod_masked = SOD.where(HH != 0).values
            sic_masked = SIC.where(HH != 0).values
            floe_masked = FLOE.where(HH != 0).values

            # --- Plot charts --- #
            fig, axs = plt.subplots(1, 3, figsize=(32, 12))

            plot_chart_data(sic_masked, 'sic', fig, axs[0],
                cm=cmo.ice, cbar_tick_spacing=2)
            plot_chart_data(sod_masked, 'sod', fig, axs[1],
                cm=cmo.deep.reversed())
            plot_chart_data(floe_masked, 'floe', fig, axs[2],
                cm=mpl.colormaps['gist_earth'])

            fig.tight_layout()
            for ax in axs:
                ax.axis('off')

            plt.suptitle(fn.split('.')[0])
            plt.savefig(out_dir + fn.split('.')[0] + '_charts.png', bbox_inches='tight')
            plt.close()

            # --- Rescale imagery to uint8 --- #
            HH_scaled = intify_band(HH, clip=True)
            HV_scaled = intify_band(HV, clip=True)
            IA_scaled = intify_band(IA, clip=False)

            # --- Make a composite --- #
            HV_denom = np.copy(HV_scaled)
            HV_denom[HV_denom==0] = 1
            ratio = np.uint8(np.float32(HH_scaled) / np.float32(HV_denom))
            ratio[HH==0] = 128
            composite = np.array([HH_scaled, HV_scaled, ratio]).transpose((1, 2, 0))

            # --- Convert to PIL.Image --- #
            HH_img = Image.fromarray(HH_scaled, mode='L')
            HV_img = Image.fromarray(HV_scaled, mode='L')
            IA_img = Image.fromarray(IA_scaled, mode='L')
            composite = Image.fromarray(composite, mode='RGB')

            HH_img.save(out_dir + fn.split('.')[0] + '_HH.jpg')
            HV_img.save(out_dir + fn.split('.')[0] + '_HV.jpg')
            IA_img.save(out_dir + fn.split('.')[0] + '_IA.jpg')
            composite.save(out_dir + fn.split('.')[0] + '_ratioRGB.jpg')


if __name__ == "__main__":
    main()