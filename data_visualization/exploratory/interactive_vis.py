import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import label, regionprops, find_contours
import os

def main():
    # train_data_fp = "./data_visualization/sample_data/raw_train/S1A_EW_GRDM_1SDH_20180721T120156_20180721T120256_022892_027B98_56D9_icechart_cis_SGRDIMID_20180721T1203Z_pl_a.nc"
    
    train_img_folder = "D:/AI4Arctic Data/raw_train/"
    train_img_fps = os.listdir(train_img_folder)


    for fn in train_img_fps:
        fp = 

        data = xr.open_dataset(train_data_fp)

        chart_nodata_mask = np.zeros(data.polygon_icechart.values.shape)
        chart_nodata_mask[~np.isnan(data.polygon_icechart.values)] = np.nan

        chart_indcs = np.copy(data.polygon_icechart.values)
        chart_indcs[np.isnan(chart_indcs)] = -1
        # label_image = label(chart_indcs, background=-1)

        # all_contours = []
        # for val in np.unique(label_image):
        #     contours = find_contours(label_image, level=val)
        #     all_contours += contours


        # Now add mouseover effects to indicate what each polygon is (SIC, SOD, FLOE) (TODO)
        fig, ax = plt.subplots(1)

        im = ax.imshow([
                data.nersc_sar_primary.values,
                data.nersc_sar_secondary.values,
                data.nersc_sar_primary.values/data.nersc_sar_secondary.values
            ])
        # for contour in all_contours:
        #     ax.plot(contour[:, 1], contour[:, 0], linewidth=0.3, color='red')

        # annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points")
        # annot.set_visible(False)

        # def update_annot(ind):
        #     # pos = im.get_offsets()[ind["ind"][0]]
        #     annot.xy = ind
        #     annot.set_text(ind)

        # def hover(event):
        #     vis = annot.get_visible()
        #     if event.inaxes == ax:
        #         cont, ind = im.contains(event)
        #         print(cont, ind)
        #         if cont:
        #             update_annot(ind)
        #             annot.set_visible(True)
        #             fig.canvas.draw_idle()
        #         else:
        #             if vis:
        #                 annot.set_visible(False)
        #                 fig.canvas.draw_idle()

        # # for region in regionprops(label_image):
        # #     # print(chart_indcs[region.coords[0][0], region.coords[0][1]])  # get chart label

        # fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.show()


if __name__ == "__main__":
    main()