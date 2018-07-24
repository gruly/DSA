#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=EN_TED_LIUM2_IWSLT
#SBATCH --mem 20G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:k40:1
#SBATCH --time=30-0
#SBATCH --error=/lium/raid01_b/ghannay/post-doc/IWSLT-2018/speech/EN_TED_LIUM2_Seq2Seq_crossEnt_freez.err
#SBATCH --output=/lium/raid01_b/ghannay/post-doc/IWSLT-2018/speech/EN_TED_LIUM2_Seq2Seq_crossEntfreez.out

echo Running on `hostname `


#use this script to do speech translation bay initializing the encoder with a pre-trained model and freeze its parameters



TRAIN="/lium/raid01_b/ghannay/post-doc/IWSLT-2018/data/TED_LIUM/TED_train_manifest.csv"
VAL="/lium/raid01_b/ghannay/post-doc/IWSLT-2018/data/TED_LIUM/TED_val_manifest.csv"
LABELS="/lium/raid01_b/ghannay/post-doc/IWSLT-2018/data/TED_LIUM/TED_LIUM2_labels_ubiqusY.json"
RES_DIR="/lium/raid01_b/ghannay/post-doc/IWSLT-2018/speech"
PRETRAIN="/lium/raid01_b/ghannay/post-doc/IWSLT-2018/speech/models/deep_speech_final_hidden_size_800_hidden_layers_5_rnn_type_lstm_epochs_70_augment_True_pretrained_False_exp_mode_EN_TED_LIUM2_baseline_best_wer_26.315.pth.tar"

python train_cross_entropy_.py \
  --train_manifest $TRAIN \
  --val_manifest $VAL --hidden_size 800   --exp_mode TED_LIUM2_Deep_Speech_Seq2Seq_crossEntfreez\
  --log_params --cuda --tensorboard --rnn_type lstm --augment --freeze \
  --labels_path $LABELS --pretrained $PRETRAIN \
  --wdir $RES_DIR --batch_size 20\
  --checkpoint \
  --epochs 70  


