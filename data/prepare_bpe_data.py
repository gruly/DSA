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

parser = argparse.ArgumentParser(description='Applay BPE for all the data ')
parser.add_argument("--target_dir", default='TEDLIUM_dataset/', type=str, help="Directory to store the dataset.")
parser.add_argument("--bpe_codes", default='TEDLIUM_dataset/', type=str, help=" File with BPE codes (created by learn_bpe.py)")
parser.add_argument("--source_dir", type=str, help="source dir ")
args = parser.parse_args()


def create_manifest(target_dl_dir, data_path, tag, ordered=True):
    manifest_path = '%s%s_manifest.csv' % (target_dl_dir,tag)
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
            transcript_path = wav_path.replace('/wav/', '/txt/').replace('.wav', '.bpe')
            sample = os.path.abspath(wav_path) + ',' + os.path.abspath(transcript_path) + '\n'
            file.write(sample.encode('utf-8'))
    print('\n')





def apply_bpe(src_file, tgt_file, bpe_codes ):
    subprocess.call(["python /lium/buster1/ghannay/Git/nmtpytorch/nmtpytorch/lib/subword_nmt/apply_bpe.py  -i {} -c {} -o {}".format(src_file, bpe_codes,tgt_file)], shell=True)




def prepare_dir(ted_dir,bpe_codes):
    converted_dir = os.path.join(ted_dir, "converted")
    txt_dir = os.path.join(converted_dir, "txt")
    # directories to store converted wav files and their transcriptions
    counter = 0
    entries = os.listdir(os.path.join(converted_dir, "txt"))
    for txt_file in tqdm(entries, total=len(entries)):
       file_name = txt_file.split('.txt')[0]
       txt_full_file = os.path.join(txt_dir, txt_file)
       target_txt_file = os.path.join(txt_dir, "{}.bpe".format( str(file_name)))
       apply_bpe(txt_full_file, target_txt_file, bpe_codes)
       counter += 1

def main():
    target_dl_dir = args.target_dir
    bpe_codes = args.bpe_codes
#    if not os.path.exists(target_dl_dir):
#        os.makedirs(target_dl_dir)


    train_ted_dir = os.path.join(args.source_dir, "train")
    val_ted_dir = os.path.join(args.source_dir, "dev")
    test_ted_dir = os.path.join(args.source_dir, "test")

  #  prepare_dir(train_ted_dir, bpe_codes)
    prepare_dir(val_ted_dir,bpe_codes)
    prepare_dir(test_ted_dir,bpe_codes)
    print('Creating manifests...')
    
    create_manifest(target_dl_dir, os.path.join(train_ted_dir,"converted"), 'bpe_train')
    create_manifest(target_dl_dir, os.path.join(val_ted_dir,"converted"), 'bpe_val')
    create_manifest(target_dl_dir, os.path.join(test_ted_dir,"converted"), 'bpe_test')



if __name__ == "__main__":
    main()
