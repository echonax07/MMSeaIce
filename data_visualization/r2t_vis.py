import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean.cm as cmocms

import os.path as osp
import argparse

"""
This can either be run as a command-line script with a filepath argument pointing to a .nc file
(e.g. `python r2t_vis.py {fp.nc}`) or the `run_vis` function can be imported and invoked programmatically
(e.g. from r2t_vis import run_vis; run_vis(fp))

Invocation will plot a 2x3 grid containing the imagery (HH, HV, IA) and chart labels (SIC, SOD, FLOE)
for the specified scene. Image masks are indicated in dark gray (land / beyond image footprint) and ice chart
nodata values (255 pixels not corresponding to image mask) are indicated in lighter gray.
"""


SOD_CLASSLABEL_LUT = {
    0: 'Ice Free',
    1: 'New',
    2: 'Young',
    3: 'Thin FY',
    4: 'Thick FY',
    5: 'Old',
    # 255: 'Glacier/NaN'  # keep commented
}
SIC_CLASSLABEL_LUT = {
    0: 'Ice Free',
    1: '1/10',
    2: '2/10',
    3: '3/10',
    4: '4/10',
    5: '5/10',
    6: '6/10',
    7: '7/10',
    8: '8/10',
    9: '9/10',
    10: '10/10',
    # 255: 'NaN'  # keep commented
}
FLOE_CLASSLABEL_LUT = {
    0: 'Ice Free',
    1: 'Ice Cake',
    2: 'Small Floe',
    3: 'Medium Floe',
    4: 'Big Floe',
    5: 'Vast/Giant Floe',
    6: 'Berg Ice',
    # 255: 'Pancake/Fast/NaN'  # keep commented
}


def percentile_clip(band, percentiles=[2, 98]):
    """
    Returns copy of data clipped to specified percentiles (list or scalar)
    """
    band_cp = np.copy(band)
    perc5, perc95 = np.nanpercentile(band_cp, percentiles)
    band_cp[band_cp < perc5] = perc5
    band_cp[band_cp > perc95] = perc95
    return band_cp


def plot_chart_data(data, type, fig, ax, cm, cbar_tick_spacing=1):
    """
    Plot a single ice chart variable.

    `type`: one of {'SIC', 'SOD' or 'FLOE'}

    `data`: 2D xr.DataArray or np.ndarray containing:
        - NaN over land (e.g. masked consistently with imagery)
        - Polygon labels consistent with data type (e.g. 0-10 for SIC)
        - 255 values where chart data is not available

    `cm`: matplotlib.colors.Colormap instance

    """
    type = type.lower()
    if type not in ['sic', 'sod', 'floe']:
        raise ValueError(f"Unsupported type '{type}'")

    label_LUT = None
    if type == 'sic':
        label_LUT = SIC_CLASSLABEL_LUT
    elif type == 'sod':
        label_LUT = SOD_CLASSLABEL_LUT
    elif type == 'floe':
        label_LUT = FLOE_CLASSLABEL_LUT

    # Set up discrete colormap for labels
    LUT_keys = list(label_LUT.keys())
    boundaries = []
    for k in LUT_keys:
        boundaries.append(k-0.5)
    boundaries.append(LUT_keys[-1]+0.5)
    assert len(boundaries) == len(LUT_keys) + 1

    cmap = cm.resampled(len(LUT_keys))
    cmap.set_bad('dimgray')  # landmask
    cmap.set_over((0.58, 0.58, 0.58))  # chart nodata mask
    norm = mpl.colors.BoundaryNorm(boundaries, extend='max', ncolors=len(LUT_keys)+1)

    # Format colobar
    cbar_ticks = []
    for i, k in enumerate(LUT_keys):
        if i == len(LUT_keys) - 1:
            cbar_ticks.append(k)
        if i % cbar_tick_spacing == 0:
            cbar_ticks.append(k)

    sic_tick_formatter = mpl.ticker.FuncFormatter(lambda val, loc: label_LUT[val])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    # Do plot
    im = ax.imshow(data, norm=norm, cmap=cmap)
    fig.colorbar(im, cax=cax, ticks=cbar_ticks, format=sic_tick_formatter)

    ax.set_title(type.upper())


def plot_img_band(img_data, ax, clip=False):
    cmap = plt.get_cmap('gray')
    cmap.set_bad('dimgray')
    if clip:
        img_data = percentile_clip(img_data)
    ax.imshow(img_data, cmap=cmap)


def run_vis(fpath, save_dir):
    """
    Driver function for visualizing imagery and associated charts
    """
    # --- Load the data --- #
    data = xr.open_dataset(fpath)

    HH = data.nersc_sar_primary
    HV = data.nersc_sar_secondary
    IA = data.sar_incidenceangle
    HH_masked = HH.where(HH != 0)
    HV_masked = HV.where(HV != 0)
    IA_masked = IA.where(IA != 0)

    SIC = data.SIC
    SOD = data.SOD
    FLOE = data.FLOE

    # Set land values to NaN, but keep other 255 values
    # (representing nodata values on the charts)
    land_indcs = (HH == 0) & (SIC == 255)
    sod_masked = SOD.where(~land_indcs)
    sic_masked = SIC.where(~land_indcs)
    floe_masked = FLOE.where(~land_indcs)

    # --- Invoke plotting functions --- #
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))

    plot_img_band(HH_masked[::4, ::4], axs[0, 0], clip=True)
    axs[0, 0].set_title("HH")
    plot_img_band(HV_masked[::4, ::4], axs[0, 1], clip=True)
    axs[0, 1].set_title("HV")
    plot_img_band(IA_masked[::4, ::4], axs[0, 2], clip=False)
    axs[0, 2].set_title("IA")

    plot_chart_data(sic_masked, 'SIC', fig, axs[1, 0],
                    cm=cmocms.ice, cbar_tick_spacing=2)
    plot_chart_data(sod_masked, 'SOD', fig, axs[1, 1],
                    cm=cmocms.deep.reversed())
    plot_chart_data(floe_masked, 'FLOE', fig, axs[1, 2],
                    cm=mpl.colormaps['gist_earth'])

    plt.suptitle(fpath.split('/')[-1])

    fig.tight_layout()
    for r in axs:
        for ax in r:
            ax.axis('off')

    # plt.show()
    plt.savefig(osp.join(save_dir, osp.basename(fpath).split('.')[0] + '.png'))
    # close figure and data to avoid memory leak and consumption
    data.close()
    plt.close()



def main():
    """
    Call this via command-line with a positional filepath arg  
    """
    parser = argparse.ArgumentParser(
        prog='r2t_vis',
        description='Visualize training images & labels'
    )
    parser.add_argument('filepath', help='Filepath of a training .nc scene')

    args = parser.parse_args()

    fp = args.filepath.replace('\\', '/')

    run_vis(fp)


if __name__ == "__main__":
    main()
