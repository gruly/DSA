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
import wave
import contextlib
import json
import codecs
from collections import OrderedDict



parser = argparse.ArgumentParser(description='Correct stm time ')
parser.add_argument("--data_type", default="", type=str, help="Data type speech_only/speech_and_noise")
parser.add_argument("--word_unit", default="c", type=str, help="word unit to process character(c)/BPE (bpe)")
parser.add_argument("--source_dir", default="", type=str, help="Path to the source dataset.")
parser.add_argument("--target_dir", default="", type=str, help="Path to the target label file computed from train data")
args = parser.parse_args()


def get_labels( source_dir, data_type,  corpus_parts):
    counter = 0
    
    entries = os.listdir(os.path.join(source_dir, "sph"))
    labels=["_"]
    for wav_file in tqdm(entries, total=len(entries)):
        speaker_name = wav_file.split('.sph')[0]
        wav_file_full = os.path.join(source_dir, "sph", wav_file)
        stm_file_full = os.path.join(source_dir, "stm","norm_acou_p2",data_type,"{}.stm".format(speaker_name))
	print ("stm_file_full  ", stm_file_full)
        if os.path.exists(wav_file_full) == True and  os.path.exists(stm_file_full) == True :
            with io.open(stm_file_full, "r", encoding='utf-8') as f:
            #with open(stm_file_full, "r") as f:
                 for stm_line in f:
                     if re.match ("^;;",stm_line) is None  :
                          tokens = stm_line.split()
                        #  characters = list(" ".join(t for t in tokens[6:]))
                          transcript = " ".join(t for t in tokens[6:]).strip().encode("utf-8", "ignore").decode("utf-8", "ignore")
                          if transcript != "ignore_time_segment_in_scoring": 
                              characters=list(transcript)
              		      for i in range(len(characters)):
                                  if characters[i] not in labels:
				     labels.append(characters[i])    
			   

    return labels    

def get_bpe_labels(source_file)    :
    labels=["_"]
    word_freqs=OrderedDict()
    nb=0
    with io.open(source_file, "r", encoding='utf-8') as f:
        for line in f:
	    words_in = line.strip().split(' ')
            for w in words_in:
		print (" word  ", w)
                if w not in word_freqs:
                    word_freqs[w] = 0
                    labels.append(w)
                    nb=nb+1
                word_freqs[w] += 1

    labels.append(" ")
    return labels

 

def main():
    source_dl_dir = args.source_dir
    target_dl_dir = args.target_dir
    data_type= args.data_type
    word_unit = args.word_unit
     
    # source dir
    source_train = os.path.join(source_dl_dir, "test")
    if word_unit != 'bpe' :
    	print " get labels form train data "
    	labels= get_labels(source_train, data_type,"test") 

    else : 
	labels=get_bpe_labels(source_dl_dir)
	data_type='bpe'
 

    label_file=os.path.join(target_dl_dir, "labels_FR_tel_test_data.json")
    #print label_file
    with io.open(label_file, 'w', encoding='utf8') as json_file:
         data=json.dumps(labels, ensure_ascii=False, indent=4)
         #data=json.dumps(labels,  indent=4)
         json_file.write(unicode(data))


    with open(label_file) as data_file:
         data_loaded = json.load(data_file)
    print " data loaded  ", str(" ".join(data_loaded).encode("utf8","ignore"))


if __name__ == "__main__":
    main()
