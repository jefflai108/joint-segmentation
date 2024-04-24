#!/bin/bash 

export PYTHONPATH=$PYTHONPATH:/data/sls/scratch/clai24/word-seg/joint-seg

data_dir=/data/sls/scratch/clai24/word-seg/joint-seg/data/spokencoco
expdir=exp/joint_supervised_phn_syllable_word/debug
mkdir -p $expdir

stage=0

if [ $stage -eq 0 ]; then 
    # train 
    PRED_LAYERS="6 6 6"
    PRED_LOSS_WEIGHT="1.0 0.0 0.0"

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
       --batch_size 32 --gradient_acc_steps 1 \
       --vocab_size 1025 --d_model 512 --nhead 8 --num_encoder_layers 6 --dim_feedforward 2048 \
       --dropout 0.1 --max_seq_length 512 --epochs 50 --log_interval 20 --learning_rate 1e-4 --label_smoothing_alpha 0.1 \
       --prediction_layers ${PRED_LAYERS} --prediction_loss_weights ${PRED_LOSS_WEIGHT}
fi 

if [ $stage -eq 1 ]; then 
    # inference 
    CUDA_LAUNCH_BLOCKING=1 python flickr8k_inference.py \
               --train_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_train.txt \
               --dev_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_dev.txt \
               --test_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_test.txt \
               --train_embed_file_path ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer14_embeddings_train.npz \
               --dev_embed_file_path ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer14_embeddings_dev.npz \
               --test_embed_file_path ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer14_embeddings_test.npz \
               --word_seg_file_path ${data_dir}/word_seg_features/flicker_speech_features.mat \
               --save_dir ${expdir} \
               --load_model_path exp/debug/best_loss_model.pth \
               --batch_size 1 \
               --repre_dim 1024 \
               --vocab_size 1027 --d_model 256 --nhead 8 --num_encoder_layers 5 --num_decoder_layers 3 --dim_feedforward 1024 \
               --dropout 0.1 --max_seq_length 100 --epochs 10 --log_interval 20 --learning_rate 1e-4 --optimizer_type "adam" --label_smoothing 0.1
fi 
