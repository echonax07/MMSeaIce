#!/bin/bash 
set -e

item="configs/feature_selection/HH_HV_enviromental_AMSR2_36_5.py"

wandb_project=feature_selection_base

for i in {1..40}; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train_infer.sh $item $wandb_project $seed
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 10

done
