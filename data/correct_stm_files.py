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


parser = argparse.ArgumentParser(description='Correct stm time ')
parser.add_argument("--data_type", default="norm_acou_p2", type=str, help="Data type speech_only/speech_and_noise")
parser.add_argument("--source_dir", default="", type=str, help="Path to the source dataset.")
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
args = parser.parse_args()

#log_file = open("message.log","a")
#sys.stdout =log_file

def get_utterances_from_stm(stm_file, wav_duration):
    """
    Return list of entries containing phrase and its start/end timings
    :param stm_file:
    :return:
    """
    res = []
    with io.open(stm_file, "r", encoding='utf-8') as f:
        for stm_line in f:
            if re.match ("^;;",stm_line)   is None : 
	  	      
               tokens = stm_line.split()
#	       if len(tokens) == 0:
#			return res
               start_time = float(tokens[3])
               end_time = float(tokens[4])
               if wav_duration < end_time:
                  print " end_time  ", end_time, " wav_duration ", wav_duration
		  tokens[4] = str(wav_duration)
               filename = tokens[0]
               transcript = " ".join(t for t in tokens).strip().encode("utf-8", "ignore").decode("utf-8", "ignore")
               res.append({"start_time": start_time, "end_time": end_time, "filename": filename, "transcript": transcript })
        return res
    



def _preprocess_transcript(phrase):
    return phrase.strip().lower()

def check_stm_file(source_dir,data_type,speaker_name):
    stm = os.path.join(source_dir, "stm","norm_acou_p2","stm_corr","{}.stm".format(speaker_name))
    if os.path.exists(stm) == True :
	return stm
    elif os.path.exists(os.path.join(source_dir, "stm","norm_acou_p2", "{}.suite.stm".format(speaker_name))) ==True :
	return os.path.join(source_dir, "stm","norm_acou_p2", "{}.suite.stm".format(speaker_name))
    else:
	return os.path.join(source_dir, "stm","norm_acou_p2", "{}.automatique.stm".format(speaker_name))




def getDurationFile(sPathFile):
    sPathFile = os.path.realpath(sPathFile)
    with contextlib.closing(wave.open(sPathFile,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return(duration)



def prepare_dir( source_dir,data_type):
    count=0
    entries = os.listdir(os.path.join(source_dir, "wav"))
    for wav_file in tqdm(entries, total=len(entries)):
        speaker_name = wav_file.split('.wav')[0]

	
        wav_file_full = os.path.join(source_dir, "wav", wav_file)
        stm_file_full = os.path.join(source_dir, "stm",data_type,"{}.stm".format(speaker_name))
        #stm_file_full = os.path.join(source_dir, "stm","{}.stm".format(speaker_name))
       # stm_res_file_full = os.path.join(source_dir, "stm_norm",data_type,"{}_corr.stm".format(speaker_name))
        if os.path.exists(wav_file_full) == True and  os.path.exists(stm_file_full) == True and os.path.getsize(stm_file_full) > 0 :
             
	    wav_file_duration_s=getDurationFile(wav_file_full)
	    print "  stm_file_full  ", stm_file_full
            all_utterances = get_utterances_from_stm(stm_file_full, wav_file_duration_s)
            #f=io.FileIO(stm_file_full, "w")
            last_utt=all_utterances[len(all_utterances)-1] 
            #f=io.FileIO(stm_res_file_full, "w")
            if wav_file_duration_s < last_utt["end_time"]:
               count+=1
               print " stm file ", stm_file_full, 
 	       print " wav file duration  ", wav_file_duration_s, " last_utt[ end time ]", last_utt["end_time"]    
               f=io.FileIO(stm_file_full, "w")
     	       for utterance_id, utterance in enumerate(all_utterances):
                   f.write(utterance["transcript"].encode('utf-8')+"\n")  
        else :
  	       print " Missing   stm file ", stm_file_full,  " wav_file_full ", wav_file_full

    print " nb modified files = ", count 		   
             

def main():
    source_dl_dir = args.source_dir
    data_type= args.data_type

     
    # source dir
    source_train = os.path.join(source_dl_dir, "train")
    source_val = os.path.join(source_dl_dir, "dev")
    source_test = os.path.join(source_dl_dir, "test")
 
    print " prepare data for train  \n \n"
    prepare_dir(source_train, data_type)
 #   print " prepare data for dev  \n \n"
    prepare_dir(source_val, data_type)
#    print " prepare data for test  \n \n"
    prepare_dir(source_test, data_type)



if __name__ == "__main__":
    main()
