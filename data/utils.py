from __future__ import print_function

import fnmatch
import io
import os
from tqdm import tqdm
import subprocess


def create_manifest(target_dl_dir, data_path, tag, data_type="", ordered=True):
    manifest_path = '%s%s_%s_manifest.csv' % (target_dl_dir,data_type,tag)
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


def _order_files(file_paths):
    print("Sorting files by length...")

    def func(element):
        output = subprocess.check_output(
            ['soxi -D \"%s\"' % element.strip()],
            shell=True
        )
        return float(output)

    file_paths.sort(key=func)


# SG: for NE Concept error rate evaluation 
def convert_to_NE (sent):
    map={'[':'pers', '(':'func', '{':'org', '$':'loc', '&':'prod', '%':'amount', '#':'time', ')':'event' }  
    # map each key in the sentence to each value
    for key in map:
        sent=sent.lower().replace(key, ' '+key+' ')
        sent=sent.lower().replace('  ', ' ')
	
    #process sentence and keep only NE sequence  
    l=sent.strip().split(' ')
    ne_sent=""
    for i in l:
	if map.has_key(i) :
	   ne_sent+= map[i]+' ';
    
    ne_sent.replace('  ', ' ')
    return  ne_sent

