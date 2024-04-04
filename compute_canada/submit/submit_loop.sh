#!/bin/bash 
set -e
array=(
"configs/feature_selection/remove_IA.py"
)

wandb_project=test

for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train_infer.sh ${array[i]} $wandb_project $seed
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 10

done
