#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1 # request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/home/fer96/projects/def-dclausi/AI4arctic/fer96/ai4arctic_challenge_clean/compute_canada_output/%j.out
#SBATCH --account=def-ka3scott
#SBATCH --mail-user=FernandoComputeCanada@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

module purge
module load python/3.10

echo "Loading module done"

source ~/AI4Artic2/bin/activate

echo "Activating virtual environment done"

cd $HOME/projects/def-dclausi/AI4arctic/$USER/ai4arctic_challenge_clean/


echo "starting training..."
# config=$1 
# # get the basename for the config file, basename is an inbuilt shell command
# config_basename=$(basename $config .py) 


export WANDB_MODE=offline
python quickstart.py $1 --wandb-project=$2 --seed=$3


# # the above python script will generate a .env at the workdir/config-name/.env
# env=./work_dir/$config_basename/.env

# echo 'Reading environment file'
# # read the .env file and save them as environment variable
# while read line; do export $line; done < $env

# echo "Starting testing"
# python test_upload.py $1 $CHECKPOINT