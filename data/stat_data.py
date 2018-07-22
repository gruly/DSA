import os
import sys
import tarfile
import argparse
import subprocess
import unicodedata
import io
import re
from utils import create_manifest
from tqdm import tqdm

parser = argparse.ArgumentParser(description='compute duration, nb words and nb of utt for the data')
parser.add_argument("--source_dir", default="", type=str, help="Path to the source dataset.")
args = parser.parse_args()

def get_duration (src_sph_file):
    #subprocess.call(["soxi -D {}".format(src_sph_file)], shell=True)
    command="soxi -D {}".format(src_sph_file)
    process = subprocess.Popen(command.split(),stdout=subprocess.PIPE)
    output, error = process.communicate()
    return float(output.strip())


def get_nb_words (src_txt_file):
    line=""
    with open(src_txt_file) as f:
    	line = f.readline()
    return (len(line.split(" ")))
    



def prepare_dir(source_dir):
    
 #   if os.path.exists(os.path.join(source_dir, "wav")) == True :
  #     convertarget_dir = source_dir
   # elif os.path.exists(os.path.join(source_dir, "converted_o","wav")) == True :
    #   convertarget_dir = os.path.join(source_dir, "converted_o")
    #else :
    convertarget_dir = os.path.join(source_dir, "converted")
    # directories to store converted wav files and their transcriptions
    nb_utt = 0
    duration=0
    nb_words=0
    print (" convertarget_dir  ", convertarget_dir)
    entries = os.listdir(os.path.join(convertarget_dir, "wav_u"))
    for sph_file in tqdm(entries, total=len(entries)):
        speaker_name = sph_file.split('.wav')[0]

        wav_file_full = os.path.join(convertarget_dir,"wav_u", "{}.wav".format(speaker_name))
        stm_file_full = os.path.join(convertarget_dir,"txt_u", "{}.txt".format(speaker_name))
        duration += get_duration(wav_file_full)
        nb_utt+=1
        nb_words += get_nb_words(stm_file_full)
#        print (" stm_file_full ", stm_file_full, " nb_utt ", nb_utt, " nb_words ", nb_words)
    print (" duration  ", round(duration/3600,1) , " nb_utt ", nb_utt, " nb_words ", nb_words)
        #if get_duration(wav_file_full)  == get_duration(sph_file_full) :


def main():
    source_dl_dir = args.source_dir
     
    # source dir
    source_train = os.path.join(source_dl_dir, "train")
    source_val = os.path.join(source_dl_dir, "dev")
    source_test = os.path.join(source_dl_dir, "test")
 
    print " prepare data for train  "
    prepare_dir(source_train) 
    print " prepare data for dev  "
#    prepare_dir(source_val)
    print " prepare data for test  "
 #   prepare_dir(source_test)



if __name__ == "__main__":
    main()
