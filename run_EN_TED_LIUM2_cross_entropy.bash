#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=EN_TED_LIUM2_IWSLT
#SBATCH --mem 20G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:k40:1
#SBATCH --time=30-0
#SBATCH --error=/lium/raid01_b/ghannay/post-doc/IWSLT-2018/speech/EN_TED_LIUM2_Seq2Seq_crossEnt.err
#SBATCH --output=/lium/raid01_b/ghannay/post-doc/IWSLT-2018/speech/EN_TED_LIUM2_Seq2Seq_crossEnt.out

echo Running on `hostname `

TRAIN="/lium/raid01_b/ghannay/post-doc/IWSLT-2018/data/TED_LIUM/TED_train_manifest.csv"
VAL="/lium/raid01_b/ghannay/post-doc/IWSLT-2018/data/TED_LIUM/TED_val_manifest.csv"
LABELS="/lium/raid01_b/ghannay/post-doc/IWSLT-2018/data/TED_LIUM/TED_LIUM2_labels_ubiqusY.json"
RES_DIR="/lium/raid01_b/ghannay/post-doc/IWSLT-2018/speech"

python train_cross_entropy.py \
  --train_manifest $TRAIN \
  --val_manifest $VAL --hidden_size 800   --exp_mode TED_LIUM2_Deep_Speech_Seq2Seq_crossEnt\
  --log_params --cuda --tensorboard --rnn_type lstm --augment \
  --labels_path $LABELS \
  --wdir $RES_DIR --batch_size 20\
  --checkpoint \
  --epochs 70
















