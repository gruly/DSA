import os
import fnmatch
import wget
import tarfile
import argparse
import subprocess
import unicodedata
import io
from utils import _order_files
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Processes and downloads TED-LIUMv2 dataset.')
parser.add_argument("--target_dir", default='TEDLIUM_dataset/', type=str, help="Directory to store the dataset.")
parser.add_argument("--source_dir", type=str, help="source dir ")
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
args = parser.parse_args()

def create_manifest(data_path , target_dl_dir, tag, ordered=True):
    manifest_src_path = '%s%s_src_manifest.csv' % (target_dl_dir,tag)
    manifest_tgt_path = '%s%s_tgt_manifest.csv' % (target_dl_dir,tag)
    print (" target_dl_dir  ", target_dl_dir)
    print (" data_path  ", data_path)
    file_paths = []
    wav_files = [os.path.join(dirpath, f)
                 for dirpath, dirnames, files in os.walk(data_path)
                 for f in fnmatch.filter(files, '*.wav')]
 #   print ("  wav_files ", wav_files)
    for file_path in tqdm(wav_files, total=len(wav_files)):
        file_paths.append(file_path.strip())
    print('\n')
    if ordered:
        _order_files(file_paths)
    file_src= io.FileIO(manifest_src_path, "w")
    file_tgt= io.FileIO(manifest_tgt_path, "w")
    for wav_path in tqdm(file_paths, total=len(file_paths)):
            transcript_path = wav_path.replace('/wav/', '/txt/').replace('.wav', '.txt')
            wav = os.path.abspath(wav_path) + '\n'
            txt = os.path.abspath(transcript_path) + '\n'
            file_src.write(wav.encode('utf-8'))
            file_tgt.write(txt.encode('utf-8'))
    print('\n')



def _preprocess_transcript(txt_file):
    """
    Return a preprocessed transcript based on the Onmt format
    :param txt_file:
    :return:
    """
    res = []
    print (" txt_file ", txt_file)
    with io.open(txt_file, "r", encoding='utf-8') as f:
        for txt_line in f:
  #          print (" txt_line  ", txt_line)
            tokens = txt_line.split(" ")
   #         print ("token  ", tokens, " len ", len(tokens))
            trans=[]
            for i in range(len(tokens)):
                trans.append(" ".join(list(tokens[i].encode("utf-8", "ignore").decode("utf-8", "ignore"))))
                trans.append("<space>")

    return (" ".join(trans))       


def prepare_dir(ted_dir,target_dl_dir, data_type):
    converted_dir = os.path.join(target_dl_dir,"ls_database" ,data_type)
    if not os.path.exists(converted_dir):
        os.makedirs(converted_dir)
    txt_dir = os.path.join(converted_dir, "txt")
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    # directories to store converted wav files and their transcriptions
    counter = 0
    entries = os.listdir(os.path.join(ted_dir, "txt"))
    for txt_file in tqdm(entries, total=len(entries)):
       file_name = txt_file.split('.txt')[0]
       txt_file_full = os.path.join(ted_dir, "txt", txt_file)
       target_txt_file = os.path.join(txt_dir, "{}.txt".format( str(file_name)))
       print (" target_txt_file ",target_txt_file )
       prep_sent=_preprocess_transcript(txt_file_full)
       with io.FileIO(target_txt_file, "w") as f:
                f.write(prep_sent.encode('utf-8'))
       counter += 1
def main():
    target_dl_dir = args.target_dir
    if not os.path.exists(target_dl_dir):
        os.makedirs(target_dl_dir)


    train_ted_dir = os.path.join(args.source_dir, "train")
    val_ted_dir = os.path.join(args.source_dir, "dev")
    test_ted_dir = os.path.join(args.source_dir, "test")

   # prepare_dir(train_ted_dir, target_dl_dir,"train")
   # prepare_dir(val_ted_dir,target_dl_dir,"dev")
    #prepare_dir(test_ted_dir,target_dl_dir,"test")
    print('Creating manifests...')

    create_manifest(os.path.join(target_dl_dir,"ls_database", "train"),target_dl_dir, 'ted_train')
    create_manifest(os.path.join(target_dl_dir, "ls_database","dev"),target_dl_dir, 'ted_val')
    create_manifest(os.path.join(target_dl_dir, "ls_database","test"),target_dl_dir, 'ted_test')



if __name__ == "__main__":
    main()
