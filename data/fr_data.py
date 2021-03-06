import os
import sys
import tarfile
import argparse
import subprocess
import unicodedata
import io
import fnmatch
import re
from utils import  _order_files
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Processes French data  dataset.')
parser.add_argument("--target_dir", default='FR_dataset/', type=str, help="Directory to store the processed dataset.")
parser.add_argument("--data_type", default="norm_ne_p2", type=str, help="Data type speech_only/speech_and_noise")
parser.add_argument("--source_dir", default="", type=str, help="Path to the source dataset.")
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
args = parser.parse_args()

#log_file = open("message.log","a")
#sys.stdout =log_file



def create_manifest(target_dl_dir, data_path, tag, data_type="", ordered=True):
    manifest_path = '%s%s_%s_manifest_data.csv' % (target_dl_dir,data_type,tag)
    print (" data_path  ", data_path)
    print (" target_dl_dir  ", target_dl_dir)
    file_paths = []
    wav_files = [os.path.join(dirpath, f)
                 for dirpath, dirnames, files in os.walk(data_path)
                 for f in fnmatch.filter(files, '*.wav')]
    for file_path in tqdm(wav_files, total=len(wav_files)):
        file_paths.append(file_path.strip())
    print('\n')
    if ordered:
        _order_files(file_paths)
    with io.FileIO(manifest_path, "w") as file:
        for wav_path in tqdm(file_paths, total=len(file_paths)):
            transcript_path = wav_path.replace('/wav/', '/txt/').replace('.wav', '.txt')
            sample = os.path.abspath(wav_path) + ',' + os.path.abspath(transcript_path) + '\n'
            file.write(sample.encode('utf-8'))
    print('\n')



def get_utterances_from_stm(stm_file):
    """
    Return list of entries containing phrase and its start/end timings
    :param stm_file:
    :return:
    """
    res = []
    with io.open(stm_file, "r", encoding='utf-8') as f:
        for stm_line in f:
            if re.match ("^;;",stm_line) is None : 
               tokens = stm_line.split()
               start_time = float(tokens[3])
               end_time = float(tokens[4])
               filename = tokens[0]
	       if tokens[2] != "inter_segment_gap":
                  transcript = " ".join(t for t in tokens[6:]).strip().encode("utf-8", "ignore").decode("utf-8", "ignore")
                  if transcript != "ignore_time_segment_in_scoring" and transcript.strip() !="": # if the transcription not empty and not equal to ignore_time_segment_in_scoring
                     res.append({"start_time": start_time, "end_time": end_time, "filename": filename, "transcript": transcript   })
        return res
    

def cut_utterance(src_wav_file, target_wav_file, start_time, end_time, sample_rate=16000):
    #print " src_wav_file ", src_wav_file, " start_time ", start_time,  "  end_time  ", end_time
    subprocess.call(["sox {}  -r {} -b 16 -c 1 {} trim {} ={}".format(src_wav_file, str(sample_rate),
                                                                      target_wav_file, start_time, end_time)],
                    shell=True)


def _preprocess_transcript(phrase):
    #return phrase.strip().upper()
    return phrase.strip().lower()

def check_stm_file(source_dir,speaker_name,data_type):
    stm = os.path.join(source_dir, "stm_ne_neuroNlp",data_type, "{}.stm".format(speaker_name))
    if os.path.exists(stm) == True :
	return stm
    elif os.path.exists(os.path.join(source_dir, "stm_ne_neuroNlp",data_type, "{}.suite.stm".format(speaker_name))) ==True :
	return os.path.join(source_dir, "stm_ne_neuroNlp",data_type, "{}.suite.stm".format(speaker_name))
    else:
	return os.path.join(source_dir, "stm_ne_neuroNlp",data_type, "{}.automatique.stm".format(speaker_name))


def filter_short_utterances(utterance_info, min_len_sec=1.0):
    return utterance_info["end_time"] - utterance_info["start_time"] > min_len_sec


def prepare_dir(target_dir, source_dir,data_type):
    #convertarget_dir = os.path.join(target_dir, data_type, "converted")
    convertarget_dir = os.path.join(target_dir, "converted_aug")
    # directories to store converted wav files and their transcriptions
    wav_dir = os.path.join(convertarget_dir, "wav")
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
    txt_dir = os.path.join(convertarget_dir, "txt")
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    counter = 0
    entries = os.listdir(os.path.join(source_dir, "wav"))
    for wav_file in tqdm(entries, total=len(entries)):
        speaker_name = wav_file.split('.wav')[0]

	
        wav_file_full = os.path.join(source_dir, "wav", wav_file)
        #stm_file_full = check_stm_file(source_dir,speaker_name)
        stm_file_full = check_stm_file(source_dir,speaker_name,data_type)
	print ("stm_file_full" , stm_file_full)
        if os.path.exists(wav_file_full) == True and  os.path.exists(stm_file_full) == True :
            all_utterances = get_utterances_from_stm(stm_file_full)


	    print " stm_file_full, ",stm_file_full
            all_utterances = filter(filter_short_utterances, all_utterances)
        
            for utterance_id, utterance in enumerate(all_utterances):
                target_wav_file = os.path.join(wav_dir, "{}_{}.wav".format(utterance["filename"], str(utterance_id)))
                target_txt_file = os.path.join(txt_dir, "{}_{}.txt".format(utterance["filename"], str(utterance_id)))
                cut_utterance(wav_file_full, target_wav_file, utterance["start_time"], utterance["end_time"],
                          sample_rate=args.sample_rate)
                with io.FileIO(target_txt_file, "w") as f:
                     f.write(_preprocess_transcript(utterance["transcript"]).encode('utf-8'))
            counter += 1
	else:
	    
	    print " stm_file_full ", stm_file_full, " wav_file_full " , wav_file_full

def main():
    target_dl_dir = args.target_dir
    source_dl_dir = args.source_dir
    #speech_only/speech_and_noise
    data_type= args.data_type
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
    #prepare_dir(target_train_dir, source_train, data_type) 
    print " prepare data for dev  "
    #prepare_dir(target_val_dir, source_val, data_type)
    print " prepare data for test  "
    prepare_dir(target_test_dir, source_test, data_type)
    print('Creating manifests...')
    print " target_train_dir ", target_train_dir 
    print " target_dl_dir  ", target_dl_dir
    #create_manifest(target_dl_dir, os.path.join(target_train_dir, "converted"), 'train',data_type)
    #create_manifest(target_dl_dir, os.path.join(target_val_dir, "converted"), 'val',data_type)
    create_manifest(target_dl_dir, os.path.join(target_test_dir, "converted"), 'test', data_type)


if __name__ == "__main__":
    main()
