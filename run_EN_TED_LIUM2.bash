#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=EN_TED_LIUM2_IWSLT
#SBATCH --mem 20G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:k40:5
#SBATCH --time=30-0
#SBATCH --error=EN_TED_LIUM2_baseline.err
#SBATCH --output=EN_TED_LIUM2_baseline.out

echo Running on `hostname `

TRAIN="/lium/raid01_b/ghannay/post-doc/IWSLT-2018/data/TED_LIUM/TED_train_manifest.csv"
VAL="/lium/raid01_b/ghannay/post-doc/IWSLT-2018/data/TED_LIUM/TED_val_manifest.csv"
LABELS="TED_LIUM2_labels_ubiqusY.json"
RES_DIR="/lium/buster1/ghannay/deepSpeech2/deepspeech_Seq2Seq"

python train.py \
  --train_manifest $TRAIN \
  --val_manifest $VAL --hidden_size 600  \
  --log_params --cuda --tensorboard --rnn_type lstm --augment \
  --labels_path $LABELS \
  --wdir $RES_DIR \
  --checkpoint \
  --epochs 70  


