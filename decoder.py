#!/usr/bin/env python
# -*- coding: <UTF-8> -*-


# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Modified to support pytorch Tensors

import editdistance as edit
import Levenshtein as Lev
import torch
from six.moves import xrange
import re

class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Arguments:
        labels (string): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
        space_index (int, optional): index for the space ' ' character. Defaults to 28.
    """

    def __init__(self, labels, blank_index=0):
        # e.g. labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
        self.labels = labels
#	print (" self.labels ", self.labels)
        self.int_to_char = dict([(i, c.lower()) for (i, c) in enumerate(labels)])
#	print (" self.int_to_char ",self.int_to_char)
        self.blank_index = blank_index
        space_index = len(labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index


    def clean (self, input):
        return(' '.join(re.sub(r'\^|\*|\<|\>|\#|\||\~', '', input).split()))
    
    def wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
#	print ("W avant s1  ",s1)
#	print ("W avant s2  ",s2)
        s1=s1.lower().replace('  ', ' ')
        s2=s2.lower().replace('  ', ' ')
	
#	print ("W apres  s1  ",s1)
#	print ("W apres  s2  ",s2)
        # build mapping of words to integers
        b = set(s1.split() + s2.split())
      #  print "s1  ", s1, " s2 ",s2 , " b ", b
        word2char = dict(zip(b, range(len(b))))
      #  print "word2char  ",word2char

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
      #  print " w1  ", w1
        w2 = [chr(word2char[w]) for w in s2.split()]
      #  print " w2  ", w2
        return Lev.distance(''.join(w1), ''.join(w2))

    def cer(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        s1=s1.lower()
        s2=s2.lower()
#	print (" C avant s1  ",s1)
        s1, s2, = s1.replace(' ',''), s2.replace(' ','')
#	print (" C  apres  s1  ",s1)
#        print " s1--",  s1 , "s2 ", s2
 #       if type(s1) != type(s2):
#		print " diff type "
        s1=unicode(s1)
        s2=unicode(s2)
        #print " s1--",  s1 , "s2 ", s2,  Lev.distance(s1, s2)," ", edit.eval(s1,s2)
        return Lev.distance(s1, s2)

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError


class BeamCTCDecoder(Decoder):
    def __init__(self, labels, lm_path=None, alpha=0, beta=0, cutoff_top_n=40, cutoff_prob=1.0, beam_width=100,
                 num_processes=4, blank_index=0):
        super(BeamCTCDecoder, self).__init__(labels)
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")
	print (" labels ", labels)
        self._decoder = CTCBeamDecoder(labels, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width,
                                       num_processes, blank_index)

    def convert_to_strings(self, out, seq_len):
        results = []
      #  print " leni out  ", len (out ), "len out[0]  ", len(out[0])
        for b, batch in enumerate(out):
           # print " b ",b, " batch ", batch
            utterances = []
            for p, utt in enumerate(batch):
               # print "parcour batch  b", b , " p ", p, 
                size = seq_len[b][p]
                if size > 0:
                    
                   # print " size  seq_len[b][p]  ", size , " utt[0:size]  ", utt[0:size]
                   # print  " utt[0:size]  ", utt[0:size]
                    transcript = ''.join(map(lambda x: self.int_to_char[x], utt[0:size]))
                    #print  " transcript  ", transcript
                else:
                    transcript = ''
                utterances.append(transcript)
            #print " utterances ", utterances
            results.append(utterances)
        return results
    def get_char_prob (self, probs, offsets, out,seq_len):

        results=[]
	for b, batch in enumerate(out):
	    p_char_prob=[]
            for p, utt in enumerate(batch):
                size = seq_len[b][p]
                if size > 0:
		   ut=utt[0:size]
                   utt_char_prob=[]
		   for i in range (len (ut)):
		       seq_numb=offsets[b][p][i]
		       char_id=ut[i]	
		       prob_char=probs[b][seq_numb][char_id]
		       utt_char_prob.append(prob_char)
		   p_char_prob.append(torch.FloatTensor(utt_char_prob))
                else:
		   p_char_prob.append(torch.FloatTensor())
	    results.append(p_char_prob)	
	return results	

    def convert_tensor(self, offsets, sizes):
        results = []
        for b, batch in enumerate(offsets):
            utterances = []
            for p, utt in enumerate(batch):
                size = sizes[b][p]
                if sizes[b][p] > 0:
                    utterances.append(utt[0:size])
                else:
                    utterances.append(torch.IntTensor())
            results.append(utterances)
        return results

    def decode(self, probs, sizes=None):
        """
        Decodes probability output using ctcdecode package.
        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes: Size of each sequence in the mini-batch
        Returns:
            string: sequences of the model's best guess for the transcription
        """
        probs = probs.cpu().transpose(0, 1).contiguous()
        out, scores, offsets, seq_lens = self._decoder.decode(probs)
        
#	print " \n ----------------------------------------------------------------------------- \n"
#	print " output, ", out ," ,scores,", scores ,", timesteps, ", offsets, "  out_seq_len ",seq_lens
#	print " \n ----------------------------------------------------------------------------- \n"
        strings = self.convert_to_strings(out, seq_lens)
        offsets = self.convert_tensor(offsets, seq_lens)
	char_probs = self.get_char_prob(probs, offsets,out, seq_lens)
        #print " offsets  ", offsets," \n char_probs ", char_probs, " strings ", strings
        #print  " strings ", strings
        return strings, offsets, char_probs, scores, seq_lens

class GreedyDecoder(Decoder):
    def __init__(self, labels, blank_index=0):
        super(GreedyDecoder, self).__init__(labels, blank_index)

    def convert_to_strings(self, sequences, sizes=None, remove_repetitions=False, return_offsets=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None
        
        for x in xrange(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
 #           print (" sequences[x]  ", sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])  # We only return one path
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self, sequence, size, remove_repetitions=False):
        string = ''
        offsets = []
        for i in range(size):
            char = self.int_to_char[sequence[i]]
#	    print (" int_to_char  ", self.int_to_char )
            if char != self.int_to_char[self.blank_index]:
                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1]]:
                    pass
                elif char == self.labels[self.space_index]:
#		    print (" char dans labels ", char )
                    string += ' '
                    offsets.append(i)
                else:
                    string = string + char
                    offsets.append(i)
        return string, torch.IntTensor(offsets)
        #return string, torch.IntTensor(offsets)

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of seq_length x batch x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted

        SG: comments:
		- b: a float tensor of the max probabilities
		- max_probs: the index of the retained character
		-- with this approach I can get only the one best
		 
        """
        b, max_probs = torch.max(probs.transpose(0, 1), 2)
#	print "b ", b
        strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes,
                                                   remove_repetitions=True, return_offsets=True)

#	print " offsets  ", offsets, " max_probs ", max_probs
        return strings, offsets
