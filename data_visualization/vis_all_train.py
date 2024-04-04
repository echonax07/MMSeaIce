import os
import argparse

from r2t_vis import run_vis
from tqdm import tqdm


"""
Command-line script that loops through every scene in specified directory,
plotting one at a time.

Usage:   python vis_all_train.py {dir}

"""

def main():
    parser = argparse.ArgumentParser(
        prog='vis_all_train',
        description='Plot all .nc scenes in directory specified, one at a time.'
    )
    parser.add_argument('dir', help='Directory of scenes to plot')
    parser.add_argument('save_dir', help='Save Directory of plotted scenes to')

    args = parser.parse_args()
    img_folder = args.dir.replace('\\', '/')
    if not img_folder[-1] == '/': img_folder += '/'

    for fn in tqdm(os.listdir(img_folder)):
        if fn.split('.')[-1] != 'nc': continue
        fp = img_folder + fn

        print(f"Plotting {fn}...")
        run_vis(fp,args.save_dir)


if __name__ == "__main__":
    main()