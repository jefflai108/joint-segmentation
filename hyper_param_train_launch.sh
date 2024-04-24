#!/bin/bash 

d_model=$1
num_encoder_layers=$2
ffn_dim=$3
dp=$4
lr=$5
labelsmooth=$6
pred_layers=$7
pred_loss_weights=$8

for label_smooth_alpha in 0.0 0.1 0.2; do 
for lr in 1e-4 3e-4 5e-4 7e-4; do 
for dp in 0.1 0.2 0.3; do 
    sbatch scripts/launch.slurm 512 6 2048 ${dp} ${lr} ${label_smooth_alpha} "6 6 6" "1.0 0.0 0.0"

    #sbatch scripts/launch.slurm 512 6 2048 ${dp} ${lr} ${label_smooth_alpha} "2 4 6" "0.25 0.25 0.5"
    #sbatch scripts/launch.slurm 512 6 2048 ${dp} ${lr} ${label_smooth_alpha} "2 4 6" "0.33 0.33 0.34"
    #sbatch scripts/launch.slurm 512 6 2048 ${dp} ${lr} ${label_smooth_alpha} "2 4 6" "0.5 0.25 0.25"
    #sbatch scripts/launch.slurm 512 6 2048 ${dp} ${lr} ${label_smooth_alpha} "2 4 6" "0.5 0.0 0.5"

    #sbatch scripts/launch.slurm 512 6 2048 ${dp} ${lr} ${label_smooth_alpha} "4 5 6" "0.25 0.25 0.5"
    #sbatch scripts/launch.slurm 512 6 2048 ${dp} ${lr} ${label_smooth_alpha} "4 5 6" "0.33 0.33 0.34"
    #sbatch scripts/launch.slurm 512 6 2048 ${dp} ${lr} ${label_smooth_alpha} "4 5 6" "0.5 0.25 0.25"
    #sbatch scripts/launch.slurm 512 6 2048 ${dp} ${lr} ${label_smooth_alpha} "4 5 6" "0.5 0.0 0.5"
done; done; done; 
