# AI4ArcticSeaIceChallenge

## Dependencies
The following packages and versions were used to develop and test the code along with the dependancies installed with pip:
- python==3.9.11
- jupyterlab==3.4.5
- xarray==2022.10.0
- h5netcdf
- numpy
- matplotlib
- torch==1.12.0
- tqdm==4.64.1
- sklearn
- jupyterlab
- ipywidgets
- icecream
- opencv-python-headless
- mmcv==1.7.1
- wandb

## Cloning the repo:

Clone this repo by using `git clone <link_to_repo>`

## Create a new environment

COmpute canada does not have support for Conda environments. So we'll use the inbuilt [venv](https://docs.python.org/3/library/venv.html) module to create new environments.

The repo contains a [create_env.sh](create_env.sh) which will create a virtual environment for you in **compute canada**.

To create a new environment ` bash create_env.sh <envname>`.
<br/> This will create a new env in the `~/<envname>` folder, which is nothing but root folder.

## Activating the environment

To activate the env, use the command `source ~/<envname>/bin/activate`. 

# Running the Code

Running the code in compute canada can be done in two ways.
 1. Running the program interactively
 2. Runnin the program as a job

 We usually use the 1st method to test our code and some small visulization and the 2nd is to train the actual model

## Requesting Interactive job

To request the interactive job, run the following command


`salloc --nodes 1 --time=2:30:0 --tasks-per-node=1 --mem=32G --account=def-dclausi --gpus-per-node=v100l:1 --cpus-per-task=6`

This will request a machine with 32 GB ram, 6 cpu cores, 1 Tesla v100 GPU for 2hours and 30 minutes.

It may take some time for you to get resource allocation (play some pingpong meanwhile). After you get your resource allocated.

Run the following command

```sh
module purge
module load python/3.9.6
source ~/<envname>/bin/activate
```

Now everthing is ready. Now it'll be like you're running the programs on your local computer, ofcourse there will be no GUI.

## Submitting a Job

TODO:

# Training the model

We modfified the code developed by the compettition organisers and added bunch of features such as follows
## Preparing the config file

The config file is file in which all the configuration of training the model is specified. An eg. of config file is [setup1.py](configs/feature_variation/setup1.py)

The line `_base_ = ['../_base_/base.py']` specificifies the base config file to use. 

Base config file stores all the default configuration and then a user can override the default configuration by creating a file like [setup1.py](configs/feature_variation/setup1.py). 

This concept (code also) was stolen from [mmcv config](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html) file structure. 

## Running quickstart.py

To run quickstart.py 

Run `quickstart.py <path_to_config_file.py> --wandb-project=<name_of_wandb_project> --workdir <path_to_a_folder>`

This will take all the configurations from `path_to_config_file.py` and will log the run in the `<name_of_wandb_project>` on wandb.

If `--workdir` is not specified, by default it will save the **model checkpoint**, **upload package** and **Inference on test images** in `workdir/config_file_name` <br>
else it will save everything in the specified folder.


## Running test_upload.py

To run test_upload.py

Run `test_upload.py <path_to_config_file.py> <path to pytorch checkpoint file.pth>` 



## Data visualization
### Dependencies
- cmocean

### Usage
All visualization code in this repository (see also `vip_ai4arctic/visualization` repo) is in the `data_visualization` directory.

The `vis_single_scene.ipynb` notebook provides an example visualization and link to the plotting function.

#### Visualize imagery & charts for a single scene (from NetCDF):
`python r2t_vis.py {filepath.nc}`

#### Visualize imagery & charts for all scenes in a directory (from NetCDF):
`python vis_all_train.py {dir}`

#### Export imagery & charts from NetCDF to file:
`python export_data.py {in_dir} {out_dir}`
