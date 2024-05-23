#!/bin/bash 
set -e

item="configs/feature_selection/HH_HV_only.py"

wandb_project=project
for i in {1..10}; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train_infer.sh $item $wandb_project 
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 10

done
