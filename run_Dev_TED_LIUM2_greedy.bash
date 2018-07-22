#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=GR_Dev_Deep_speech_TED_LIUM3_ubiqusY
#SBATCH --mem 50G
#SBATCH --gres=gpu:k40:1
#SBATCH --time=10-0
#SBATCH --error=GR_Dev_deepSpeech_ted3_ubiqusY.err
#SBATCH --output=GR_Dev_deepSpeech_ted3_ubiqusY.out

echo Running on `hostname `

VAL="/lium/raid01_b/ghannay/post-doc/IWSLT-2018/data/TED_LIUM/TED_train_manifestp.csv"
#LM="/lium/lst_b/yannick/asr/deep_speech/cantab-TEDLIUM-pruned.mmap"
model="/lium/buster1/ghannay/deepSpeech2/deepspeech_Seq2Seq/models/deep_speech_final_hidden_size_600_hidden_layers_5_rnn_type_lstm_epochs_70_augment_True_pretrained_False_labels_path_TED_LIUM2_labels_ubiqusY.json_best_wer_124.166666667.pth.tar"

python test.py \
  --test_manifest $VAL \
  --cuda --decoder greedy\
  --model_path $model \
  --save_path res






