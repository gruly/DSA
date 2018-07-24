#!/usr/bin/env python
# -*- coding: <UTF-8> -*-
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import time
import datetime
import random

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            batch_size = input_.size()[0]
            return torch.stack([F.softmax(input_[i]) for i in range(batch_size)], 0)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True, last=0):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)
        self.num_directions = 2 if bidirectional else 1

        self.last=last
    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        self.flatten_parameters()
        if self.last==0:
            x,_ = self.rnn(x) # output,( hidden state, cell state)
	else: # last hidden 
            x,(h,c) = self.rnn(x) # output, (hidden state, cell state)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        if self.last==0:
            return x
        else:
            return x,h


class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, n_features, context):
        # should we handle batch_first=True?
        super(Lookahead, self).__init__()
        self.n_features = n_features
        self.weight = Parameter(torch.Tensor(n_features, context + 1))
        assert context > 0
        self.context = context
        self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):  # what's a better way initialiase this layer?
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        seq_len = input.size(0)
        # pad the 0th dimension (T/sequence) with zeroes whose number = context
        # Once pytorch's padding functions have settled, should move to those.
        padding = torch.zeros(self.context, *(input.size()[1:])).type_as(input.data)
        x = torch.cat((input, Variable(padding)), 0)

        # add lookahead windows (with context+1 width) as a fourth dimension
        # for each seq-batch-feature combination
        x = [x[i:i + self.context + 1] for i in range(seq_len)]  # TxLxNxH - sequence, context, batch, feature
        x = torch.stack(x)
        x = x.permute(0, 2, 3, 1)  # TxNxHxL - sequence, batch, feature, context

        x = torch.mul(x, self.weight).sum(dim=3)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'



class DeepSpeech(nn.Module):
    def __init__(self, rnn_type=nn.LSTM, labels="abc", rnn_hidden_size=768, nb_layers=5, audio_conf=None, dec_hidden_size=300, dec_n_layers=1,dec_dropout_p=0.1, bidirectional=True, context=20):
        super(DeepSpeech, self).__init__()

	print("On entre dans Model")

	# Encoder
        # model metadata needed for serialization/deserialization
        if audio_conf is None:
            audio_conf = {}
        self._version = '0.0.1'
        self._hidden_size = rnn_hidden_size
        self._hidden_layers = nb_layers
        self._rnn_type = rnn_type
        self._audio_conf = audio_conf or {}
        self._labels = labels
        self._bidirectional = bidirectional
        # Define decoder parameters
        self._dec_hidden_size = dec_hidden_size
        self._n_layers = dec_n_layers
        self._dropout_p = dec_dropout_p 


        sample_rate = self._audio_conf.get("sample_rate", 16000)
        window_size = self._audio_conf.get("window_size", 0.02)
        num_classes = len(self._labels)
	print (" num classess ", num_classes)

	# Encoder
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(0, 10)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), ),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size - 21) / 2 + 1)
        rnn_input_size *= 32

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                       bidirectional=bidirectional, batch_norm=False, last=0)
        rnns.append(('0', rnn))
        for x in range(nb_layers - 1):

            if x != nb_layers-2 :  
               rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional, last=0)
            else:
               rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional, last=1)
            rnns.append(('%d' % (x + 1), rnn))

         
        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(rnn_hidden_size, context=context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not bidirectional else None

	print("Fin init encoder")


	# Decoder
        self.embedding = nn.Embedding(num_classes,self._dec_hidden_size)
        self.dropout = nn.Dropout(self._dropout_p)
        self.attn = Attn('general', self._hidden_size, self._dec_hidden_size)
        self.gru = nn.GRU(self._dec_hidden_size+self._hidden_size, self._dec_hidden_size, self._n_layers, dropout=self._dropout_p)
        self.fc = nn.Linear(self._dec_hidden_size+self._hidden_size,num_classes)
        self.inference_softmax = InferenceBatchSoftmax()

	print("Fin init decoder")
        

        
    def encode(self, x):
        x = self.conv(x)
        sizes = x.size()
        
        
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        x, h  = self.rnns(x)


        if not self._bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x) 
        return x, h
    
 
    def decode(self, x,encoder_hidden, y):
        """
        x should be shape (batch, time, hidden dimension)
        y should be shape (batch, label sequence length)
        """
        # Prepare decoder input and outputs
        decoder_input = Variable(torch.LongTensor([1] * y.size()[1])).cuda() # padding index
        decoder_hidden = torch.autograd.Variable(torch.rand(1,y.size(1),self._dec_hidden_size)).cuda()  #torch.zeros(1, y.size(0),self._dec_hidden_size) # 1 x B x  H
        all_decoder_outputs = Variable(torch.zeros(y.size()[0], y.size(1), len(self._labels))).cuda()
        encoder_outputs=x
        for t in range(len(y) ): # max seq length on ne fait pas -1 parce qu'on n'a pas ajouter le tag fin de phrase
            last_hidden=decoder_hidden
            word_embedded = self.embedding(decoder_input)#.view(1, 1, -1)
            word_embedded=word_embedded.view(1,word_embedded.size(0),word_embedded.size(1))
	    word_embedded = self.dropout(word_embedded).contiguous()
            # Calculate attention weights and apply to encoder outputs
            attn_weights = self.attn(last_hidden, encoder_outputs)
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
            context = context.transpose(0, 1).contiguous() # 1 x B x N
            # Combine embedded input word and attended context, run through RNN
            rnn_input = torch.cat((word_embedded, context), 2)
            self.gru.flatten_parameters()
            decoder_output, decoder_hidden = self.gru(rnn_input, last_hidden)

	    # Final output layer
            context = context.squeeze(0) # B x N
            decoder_output = decoder_output.squeeze(0) # B x N
            decoder_output = torch.cat((decoder_output, context), 1)
            decoder_output = self.fc(decoder_output)
            decoder_output = self.inference_softmax(decoder_output)
            all_decoder_outputs[t] = decoder_output # Store this step's outputs
	    #Next input is current target
            decoder_input = y[t]
        
        return all_decoder_outputs


    def forward(self, x, y):
        # encoder
        encoder_outputs, encoder_hidden = self.encode(x) 
	# decoder
        y=y.transpose(0,1)
        out = self.decode(encoder_outputs, encoder_hidden, y)
        out=out.transpose(0, 1).contiguous()
        return out
    # freeze the parameters of the encoder
    def freeze_updates (self):
        child_counter = 0
        for child in self.children():
            if child_counter < 2:
                for param in child.parameters():
                    param.requires_grad = False
            child_counter += 1


       
    # Laod from pretrained model (used for speech models : load encoder prameters)
    def load_from_pretrained_file(self, path):
        old_weights = torch.load(path, map_location=lambda storage, loc: storage)['state_dict']
	print('Removing softmax layer with shape: ', old_weights.pop('fc.0.module.1.weight').shape)
	try:
	  self.load_state_dict(old_weights)
	except KeyError as ke:
	  print(ke)
	
    @classmethod
    def load_model(cls, path, cuda=False):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(rnn_hidden_size=package['hidden_size'], nb_layers=package['hidden_layers'],
                    labels=package['labels'], audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']], bidirectional=package.get('bidirectional', True), dec_hidden_size=package['dec_hidden_size'], dec_n_layers=package['dec_n_layers'], dec_dropout_p=package['dropout_p'])
        # the blacklist parameters are params that were previous erroneously saved by the model
        # care should be taken in future versions that if batch_norm on the first rnn is required
        # that it be named something else
        blacklist = ['rnns.0.batch_norm.module.weight', 'rnns.0.batch_norm.module.bias',
                     'rnns.0.batch_norm.module.running_mean', 'rnns.0.batch_norm.module.running_var']
        for x in blacklist:
            if x in package['state_dict']:
                del package['state_dict'][x]
        model.load_state_dict(package['state_dict'])

        for x in model.rnns:
            x.flatten_parameters()
        if cuda:
            model = torch.nn.DataParallel(model).cuda()
        return model

    @classmethod
    def load_model_package(cls, package, cuda=False):
        model = cls(rnn_hidden_size=package['hidden_size'], nb_layers=package['hidden_layers'],
                    labels=package['labels'], audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']], bidirectional=package.get('bidirectional', True), dec_hidden_size=package['dec_hidden_size'], dec_n_layers=package['dec_n_layers'], dec_dropout_p=package['dropout_p'])
        model.load_state_dict(package['state_dict'])
        if cuda:
            model = torch.nn.DataParallel(model).cuda()
        return model

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  cer_results=None, wer_results=None, best_wer_results=None,  avg_loss=None, meta=None):
        model_is_cuda = next(model.parameters()).is_cuda
        
        #model = model.module if model_is_cuda else model
        model = model if model_is_cuda else model
        package = {
            'version': model._version,
            'hidden_size': model._hidden_size,
            'hidden_layers': model._hidden_layers,
            'rnn_type': supported_rnns_inv.get(model._rnn_type, model._rnn_type.__name__.lower()),
            'audio_conf': model._audio_conf,
            'labels': model._labels,
            'state_dict': model.state_dict(),
            'dec_hidden_size' : model._dec_hidden_size,
            'dec_n_layers' : model._n_layers,
	    'dropout_p' : model._dropout_p,
        }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
	    package['best_wer_results'] = best_wer_results
        if meta is not None:
            package['meta'] = meta
        return package

    @staticmethod
    def get_labels(model):
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._labels if model_is_cuda else model._labels

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params

    @staticmethod
    def get_audio_conf(model):
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._audio_conf if model_is_cuda else model._audio_conf

    @staticmethod
    def get_meta(model):
        model_is_cuda = next(model.parameters()).is_cuda
        m = model.module if model_is_cuda else model
        meta = {
            "version": m._version,
            "hidden_size": m._hidden_size,
            "hidden_layers": m._hidden_layers,
            "rnn_type": supported_rnns_inv[m._rnn_type]
        }
        return meta
class Attn(nn.Module):
    def __init__(self, method, enc_hidden_size, dec_hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = enc_hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size,dec_hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size+dec_hidden_size, dec_hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, dec_hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(max_len, this_batch_size)) # SxB
        
        #if USE_CUDA: # voir comment  faire ce test
        attn_energies = attn_energies.cuda()


	hidden = hidden.transpose(1,0)

        for i in range(max_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i].unsqueeze(0))
        attn_energies=attn_energies.transpose(1,0)
        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
	   # print("energy.size()",energy.transpose(1,0).transpose(2,1).size())
	   # print("hidden.size()",hidden.size())
            energy = torch.bmm(hidden, energy.squeeze(0).unsqueeze(2))
            return energy
        
        elif self.method == 'concat':
           # print (" concat \n")
           # print (" hidden size", hidden.size(), type(hidden))   
            #print (" hidden ", hidden)   
          #  print (" encoder output size", encoder_output.size(), type(encoder_output))   
         #   print (" torch.cat((hidden, encoder_output), 1)  ", torch.cat((hidden, encoder_output), dim=1))   
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
	    
        #    print ( " energy  ", energy.size())
            energy = torch.dot(self.v.view(-1),energy.view(-1))
            return energy



