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

parser = argparse.ArgumentParser(description='Processes French data  dataset.')
parser.add_argument("--target_dir", default='FR_dataset/', type=str, help="Directory to store the processed dataset.")
parser.add_argument("--source_dir", default="", type=str, help="Path to the source dataset.")
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
args = parser.parse_args()

#log_file = open("message.log","a")
#sys.stdout =log_file



def prepare_dir(target_dir, source_dir,corpus):
    # directories to store converted wav files and their transcriptions
    txt_dir = os.path.join(target_dir, "txt")
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    counter = 0
    aligment_file=open(os.path.join(source_dir,"alignments.meta"),"r")
    aligment_file.readline()
    aligments=aligment_file.readlines()
    transcript=open(os.path.join(source_dir,"%s.en"%(corpus)),"r").readlines()
    assert len(aligments) == len(transcript) , " aligments size should be equal to transcript size" 
    for i in range (len(aligments)):
        l=aligments[i].strip().split("\t")
        trans=transcript[i].strip().lower()	
        target_txt_file = os.path.join(txt_dir, "{}.txt".format(l[4]))
        with io.FileIO(target_txt_file, "w") as f:
             f.write(trans.encode('utf-8'))

def main():
    target_dl_dir = args.target_dir
    source_dl_dir = args.source_dir
    #speech_only/speech_and_noise
   
    if not os.path.exists(target_dl_dir):
        os.makedirs(target_dl_dir)

     
    # prepare target dir
    target_train_dir = os.path.join(target_dl_dir, "train")
    if not os.path.exists(target_train_dir):
        os.makedirs(target_train_dir)
    target_val_dir = os.path.join(target_dl_dir, "dev")
    if not os.path.exists(target_val_dir):
        os.makedirs(target_val_dir)
    target_test_dir = os.path.join(target_dl_dir, "test")
    if not os.path.exists(target_test_dir):
        os.makedirs(target_test_dir)
    # source dir
    source_train = os.path.join(source_dl_dir, "train")
    source_val = os.path.join(source_dl_dir, "dev")
    source_test = os.path.join(source_dl_dir, "test")
 
    print " prepare data for train  "
    #prepare_dir(target_train_dir, source_train,"train") 
    print " prepare data for dev  "
    #prepare_dir(target_val_dir, source_val,"dev")
    print " prepare data for test  "
    #prepare_dir(target_test_dir, source_test,"test")
    print('Creating manifests...')
    print (" target_train_dir  ", target_train_dir)

    create_manifest(target_dl_dir, target_train_dir, 'train')
    create_manifest(target_dl_dir, target_val_dir, 'val')
    create_manifest(target_dl_dir, target_test_dir, 'test')


if __name__ == "__main__":
    main()
