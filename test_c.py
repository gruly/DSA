import argparse

import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import os
from decoder import GreedyDecoder
import re
from data.data_loader import SpectrogramDataset, AudioDataLoader
from data.utils import convert_to_NE 
from model import DeepSpeech

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--test_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam", "none"], type=str, help="Decoder to use")
parser.add_argument('--eval', default="word", choices=["concept", "word"], type=str, help="Evaluation of the concept in case of NE, so we keep in the sent only the NE tags")
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
no_decoder_args = parser.add_argument_group("No Decoder Options", "Configuration options for when no decoder is "
                                                                  "specified")
no_decoder_args.add_argument('--output_path', default=None, type=str, help="Where to save raw acoustic output")
beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--top_paths', default=1, type=int, help='number of beams to return')
beam_args.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm_path', default=None, type=str,
                       help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--nbest_path', default=None, type=str, help='file to save the nbest list')
beam_args.add_argument('--save_path', default=None, type=str, help='path to save the refrence and hypothesis transcriptions')
beam_args.add_argument('--alpha', default=0.8, type=float, help='Language model weight')
beam_args.add_argument('--beta', default=1, type=float, help='Language model word bonus (all words)')
beam_args.add_argument('--cutoff_top_n', default=40, type=int,
                       help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                            'vocabulary will be used in beam search, default 40.')
beam_args.add_argument('--cutoff_prob', default=1.0, type=float,
                       help='Cutoff probability in pruning,default 1.0, no pruning.')
beam_args.add_argument('--lm_workers', default=1, type=int, help='Number of LM processes to use')
args = parser.parse_args()

if __name__ == '__main__':
    model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
    model.eval()

    model_name=re.sub('.json.pth.tar','',os.path.basename(args.model_path))
    corpus=re.sub('csv','',os.path.basename(args.test_manifest))
    if args.save_path : 
        ref_file=open("%s/%s_reference.%s.txt"%( args.save_path, corpus, model_name),'w')
        trans_file=open("%s/%s_transcription.%s.txt"%(args.save_path, corpus, model_name),'w')
    labels = DeepSpeech.get_labels(model)
    print ("model_name ", model_name)
    audio_conf = DeepSpeech.get_audio_conf(model)

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder
	print ( "alpha=args.alpha, beta=args.beta ", args.alpha, args.beta)
        decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers, blank_index=labels.index('_'))
    elif args.decoder == "greedy":
        decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    else:
        decoder = None
    target_decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.test_manifest, labels=labels,
                                      normalize=True)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    total_cer, total_wer = 0, 0
    output_data = []
    total_err_c=0

    sent_ne = 0 # reference sentences that hav ne tags ( the Concept error rate will be evaluated only on those sentences)
    cp=0
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes, audio_ids  = data
	#print " audio_ids  ", audio_ids 
        inputs = Variable(inputs, volatile=True)

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        if args.cuda:
            inputs = inputs.cuda()

        out = model(inputs)
        out = out.transpose(0, 1)  # TxNxH
        seq_length = out.size(0)
        sizes = input_percentages.mul_(int(seq_length)).int()


#	print " seq_length  ", seq_length, "sizes  ", sizes 
        if decoder is None:
            # add output to data array, and continue
            output_data.append((out.data.cpu().numpy(), sizes.numpy()))
            continue

        if args.decoder == "beam" : 
            decoded_output, offsets, char_probs, scores, seq_lens = decoder.decode(out.data, sizes)
	else:
            decoded_output, _, = decoder.decode(out.data, sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)
        wer, cer = 0, 0
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
	    if args.save_path :
		ref_file.write(reference.encode('utf-8')+"("+audio_ids[x]+")\n")            
		#ref_file.write(reference.encode('utf-8')+"(sent_"+str(cp)+")\n")            
		trans_file.write(transcript.encode('utf-8')+"("+audio_ids[x]+")\n")            
		#trans_file.write(transcript.encode('utf-8')+"(sent_"+str(cp)+")\n")      
                cp=cp+1 
    
            wer_inst=0
            # SG : taking into account only the concept during evaluation 
	    if args.eval=='concept' : # evaluation of the concept : clean refrence and sentence
		 new_ref= convert_to_NE(reference)
		 if len(new_ref.split()) !=0 :
		     sent_ne+=1
		     wer_inst = decoder.wer(convert_to_NE(transcript), convert_to_NE(reference)) / float(len(new_ref.split()))
		     print (" len(new_ref.split())  ",len(new_ref.split())," wer_inst  " ,wer_inst, "decoder.wer(convert_to_NE(transcript), convert_to_NE(reference))", decoder.wer(convert_to_NE(transcript), convert_to_NE(reference)) ," len(convert_to_NE(transcript)) ", len(convert_to_NE(transcript).split()))
		# else:
		    # wer_inst=len(convert_to_NE(transcript).split())
		 total_err_c+=decoder.wer(convert_to_NE(transcript), convert_to_NE(reference)) 
	    else:
            	wer_inst = decoder.wer(transcript, reference) / float(len(reference.split()))
            cer_inst = decoder.cer(transcript, reference) / float(len(reference))
            wer += wer_inst
            cer += cer_inst
            if args.verbose:
                print("Ref:", reference.lower())
                print("Hyp:", transcript.lower())
                print("WER:", wer_inst, "CER:", cer_inst, "\n")
	    if args.nbest_path != None :
#		print " saving the ",args.beam_width , "best list "
                
		nbest_file=open(args.nbest_path,'a')
                nbest_file.write(" REF: "+ " "+reference.encode('utf-8')+"\n")
		for j in range (10):
#		    print " audio_ids ", audio_ids[x], " x j ", x, " ",j, " score  ",scores[x][j], " utterance  ", decoded_output[x][j]
		    nbest_file.write(audio_ids[x]+" "+str(scores[x][j])+" "+decoded_output[x][j].encode('utf-8')+"\n")
	        nbest_file.close()
		
        total_cer += cer
        total_wer += wer

    if decoder is not None:
	if args.eval=='concept':
            #wer = total_wer/len(test_loader.dataset)
            wer = total_wer/float(sent_ne) # dans ce score on prend en conte  que les phrases qui contiennent les etiquettes NE 
	    print (" wer ", wer , " total_wer ", total_wer, "  test_loader.dataset  ", len(test_loader.dataset), " sent ne ", sent_ne," total_err_c ", total_err_c)
	else:
       	    wer = total_wer / len(test_loader.dataset)
        cer = total_cer / len(test_loader.dataset)

	if args.eval=='concept': 
            print('Test Summary \t'
              'Average Concept Error Rate {Conper:.3f}\t'
              'Average CER {cer:.3f}\t'.format(Conper=wer * 100, cer=cer * 100))
	else:
	    print('Test Summary \t'
              'Average Concept Error Rate {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(wer=wer * 100, cer=cer * 100))

    else:
        np.save(args.output_path, output_data)
