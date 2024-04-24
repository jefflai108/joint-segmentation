#!/bin/bash 
#SBATCH -J jointseg
#SBATCH -o /data/sls/scratch/clai24/slurm_dumps/jointseg-debug-hyperparam-search_%j.out   
#SBATCH -e /data/sls/scratch/clai24/slurm_dumps/jointseg-debug-hyperparam-search_%j.err   
#SBATCH --qos=regular 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 
#SBATCH --partition=sm,2080
#SBATCH --exclude sls-2080-2,sls-sm-5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00 
#SBATCH --requeue
#SBATCH --mem=60G

## Set the python environment you want to use for your code 
CONDA_ROOT=/data/sls/scratch/clai24/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh 
conda activate /data/sls/scratch/clai24/anaconda3/envs/prompt-gpt4

export PYTHONPATH=$PYTHONPATH:/data/sls/scratch/clai24/word-seg/joint-seg

data_dir=/data/sls/scratch/clai24/word-seg/joint-seg/data/spokencoco

d_model=$1
num_encoder_layers=$2
ffn_dim=$3
dp=$4
lr=$5
labelsmooth=$6
pred_layers=$7
pred_loss_weights=$8

stage="joint_supervised_phn_syllable_word"

if [ "$stage" == "joint_supervised_phn_syllable_word" ]; then
    pred_layers_name="${pred_layers// /_}"
    pred_loss_weights_name="${pred_loss_weights// /_}"
    expdir=/data/sls/scratch/clai24/word-seg/joint-seg/exp/joint_supervised_phn_syllable_word/v0/train_split00_dmodel${d_model}_enclayer${num_encoder_layers}_ffn_dim${ffn_dim}_dp${dp}_lr${lr}_labelsmooth${labelsmooth}_predlayers${pred_layers_name}_predlossweights${pred_loss_weights_name}
    mkdir -p $expdir

    python main.py \
       --train_token_file_path ${data_dir}/speechtokens/rvq1/spokencoco_rvq1_tokens_train_split_00.txt \
       --dev_token_file_path ${data_dir}/speechtokens/rvq1/spokencoco_rvq1_tokens_dev.txt \
       --train_phone_label_file_path ${data_dir}/labels/spokencoco_train_phone_labels.npz \
       --dev_phone_label_file_path ${data_dir}/labels/spokencoco_dev_phone_labels.npz \
       --train_syllable_label_file_path ${data_dir}/labels/spokencoco_train_syllable_labels.npz \
       --dev_syllable_label_file_path ${data_dir}/labels/spokencoco_dev_syllable_labels.npz \
       --train_word_label_file_path ${data_dir}/labels/spokencoco_train_word_labels.npz \
       --dev_word_label_file_path ${data_dir}/labels/spokencoco_dev_word_labels.npz \
       --save_dir ${expdir} \
       --batch_size 32 --gradient_acc_steps 20 \
       --vocab_size 1025 --d_model ${d_model} --nhead 8 --num_encoder_layers ${num_encoder_layers} --dim_feedforward ${ffn_dim} \
       --dropout ${dp} --max_seq_length 512 --epochs 10 --log_interval 40 --learning_rate ${lr} --label_smoothing_alpha ${labelsmooth} \
       --prediction_layers ${pred_layers} --prediction_loss_weights ${pred_loss_weights} \
       >> ${expdir}/training.log 2>&1
fi 

