#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=v100:1 # request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=75G
#SBATCH --time=14:00:00
#SBATCH --output=compute_canada_output/%j.out
#SBATCH --account=def-dclausi
#SBATCH --mail-user=muhammed.computecanada@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

module purge
module load python/3.9.6

echo "Loading module done"

source ~/env7_ai4arctic/bin/activate

echo "Activating virtual environment done"

#cd /project/def-dclausi/share/whale/mmwhale/
cd $HOME/projects/def-dclausi/AI4arctic/$USER

echo "starting training..."

export WANDB_MODE=offline

python quickstart.py $1 
python test_upload.py $1 $CHECKPOINT