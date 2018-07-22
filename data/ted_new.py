import os
import wget
import tarfile
import argparse
import subprocess
import unicodedata
import io
from utils import create_manifest
from tqdm import tqdm
import contextlib
import wave

parser = argparse.ArgumentParser(description='Processes and downloads TED-LIUMv2/TED_LIUM3 dataset.')
parser.add_argument("--target_dir", default='TEDLIUM_dataset/', type=str, help="Directory to store the dataset.")
parser.add_argument("--source_dir", type=str, help="TEDLIUM ubiquis source dir.")
parser.add_argument("--csv_file", type=str, help="name of csv file")
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
args = parser.parse_args()

TED_LIUM_V2_DL_URL = "http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz"


def get_utterances_from_stm(stm_file):
    """
    Return list of entries containing phrase and its start/end timings
    :param stm_file:
    :return:
    """
    res = []
    with io.open(stm_file, "r", encoding='utf-8') as f:
        for stm_line in f:
            tokens = stm_line.split()
            start_time = float(tokens[3])
            end_time = float(tokens[4])
            filename = tokens[0]
            transcript = unicodedata.normalize("NFKD",
                                               " ".join(t for t in tokens[6:]).strip().replace("<unk>","")). \
                encode("utf-8", "ignore").decode("utf-8", "ignore")
            if transcript != "ignore_time_segment_in_scoring":
                res.append({
                    "start_time": start_time, "end_time": end_time,
                    "filename": filename, "transcript": transcript
                })
        return res


def cut_utterance(src_sph_file, target_wav_file, start_time, end_time, sample_rate=16000):
    subprocess.call(["sox {}  -r {} -b 16 -c 1 {} trim {} ={}".format(src_sph_file, str(sample_rate),
                                                                      target_wav_file, start_time, end_time)],
                    shell=True)


def _preprocess_transcript(phrase):
    return phrase.strip().lower()


def filter_short_utterances(utterance_info, min_len_sec=1.0):
    return utterance_info["end_time"] - utterance_info["start_time"] > min_len_sec

def get_duration (src_sph_file):
    #subprocess.call(["soxi -D {}".format(src_sph_file)], shell=True)
    command="soxi -D {}".format(src_sph_file)
    process = subprocess.Popen(command.split(),stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output



def prepare_dir(ted_dir):
    converted_dir = os.path.join(ted_dir, "converted")
    # directories to store converted wav files and their transcriptions
    wav_dir = os.path.join(converted_dir, "wav_u")
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
    txt_dir = os.path.join(converted_dir, "txt_u")
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    counter = 0
    entries = os.listdir(os.path.join(ted_dir, "sph"))
    for sph_file in tqdm(entries, total=len(entries)):
        speaker_name = sph_file.split('.sph')[0]

        sph_file_full = os.path.join(ted_dir, "sph", sph_file)
	print "sph_file_full ",sph_file_full
        stm_file_full = os.path.join(ted_dir, "stm_ubiqus", "{}.stm".format(speaker_name))

        assert os.path.exists(sph_file_full) and os.path.exists(stm_file_full)
        all_utterances = get_utterances_from_stm(stm_file_full)
	duration=get_duration(sph_file_full)
	print (" duration ", duration)
	s=0
        all_utterances = filter(filter_short_utterances, all_utterances)
        for utterance_id, utterance in enumerate(all_utterances):
            target_wav_file = os.path.join(wav_dir, "{}_{}.wav".format(utterance["filename"], str(utterance_id)))
            target_txt_file = os.path.join(txt_dir, "{}_{}.txt".format(utterance["filename"], str(utterance_id)))
	    if float(utterance["end_time"]) < float(duration) :
            	cut_utterance(sph_file_full, target_wav_file, utterance["start_time"], utterance["end_time"],
                	          sample_rate=args.sample_rate)
            	with io.FileIO(target_txt_file, "w") as f:
                	f.write(_preprocess_transcript(utterance["transcript"]).encode('utf-8'))
            else:
		if s==0:
			print " ==> Problem with this file ",sph_file_full 
			s=1
	counter += 1


def main():
    target_dl_dir = args.target_dir
    source_dl_dir = args.source_dir
    if not os.path.exists(target_dl_dir):
        os.makedirs(target_dl_dir)

    source_train = os.path.join(source_dl_dir, "train")

    train_ted_dir = os.path.join(target_dl_dir, "train")
   # val_ted_dir = os.path.join(target_unpacked_dir, "dev")
   # test_ted_dir = os.path.join(target_unpacked_dir, "test")

    prepare_dir(source_train)
    #prepare_dir(val_ted_dir)
    #prepare_dir(test_ted_dir)
    print('Creating manifests...')

    create_manifest(train_ted_dir, os.path.join(source_train, "converted"), args.csv_file,data_type="train")
    #create_manifest(target_dl_dir, os.path.join(source_train, "converted"), args.csv_file)
   # create_manifest(val_ted_dir, 'ted_val')
   # create_manifest(test_ted_dir, 'ted_test')


if __name__ == "__main__":
    main()
