#!/usr/bin/env python
# -*- coding: <UTF-8> -*-
import argparse
import errno
import json
import os
import time
import torch
from tqdm import tqdm
from torch.autograd import Variable
from masked_cross_entropy import *
from decoder import GreedyDecoder
from model_seq2seqAtt import DeepSpeech, supported_rnns
from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler
parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train_manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch_size', default=5, type=int, help='Batch size for training')
parser.add_argument('--exp_mode', default='', help='experiments mode: it can be the name of the corpus, type of exp')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels_path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden_size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden_layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn_type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning_anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint_per_batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log_dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log_params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--wdir', default='', help='workign directory')
parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
parser.add_argument('--pretrained', default='', help='Pretrained model file')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise_dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise_prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise_min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise_max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no_shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no_bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--dec_hidden_size', default=300, help='Decoder hidden size')
parser.add_argument('--dec_n_layers', default=1, help='Decoder number of layers')
parser.add_argument('--dec_dropout', default=0.1, help='Decoder dropout probability')


def to_np(x):
    return x.data.cpu().numpy()




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    args = parser.parse_args()
    save_folder = args.save_folder
    labels_path=os.path.basename(args.labels_path)

    log_dir_w="%s/logs"%(args.wdir)
    try:
            os.makedirs(log_dir_w)
    except OSError as e:
            if e.errno == errno.EEXIST:
                print('logs dir exists')

    model_dir="%s/%s"%(args.wdir,args.save_folder)
    try:
            os.makedirs(model_dir)
    except OSError as e:
            if e.errno == errno.EEXIST:
                print('model dir exists')

    pretrained=False
    if args.pretrained :
	pretrained=True
    #log_file =open("logs/run_deep_speech_final_hidden_size_%s_hidden_layers_%s_rnn_type_%s_epochs_%s_augment_%s.log"%(args.hidden_size,args.hidden_layers,args.rnn_type,args.epochs,args.augment),'w')
    log_file_name ="%s/run_deep_speech_final_hidden_size_%s_hidden_layers_%s_rnn_type_%s_epochs_%s_augment_%s_pretrained_%s_exp_mode_%s.log"%(log_dir_w, args.hidden_size,args.hidden_layers,args.rnn_type,args.epochs,args.augment, pretrained,args.exp_mode)
    args.model_path="%s/deep_speech_final_hidden_size_%s_hidden_layers_%s_rnn_type_%s_epochs_%s_augment_%s_pretrained_%s_exp_mode_%s.pth.tar"%(model_dir,args.hidden_size,args.hidden_layers,args.rnn_type,args.epochs,args.augment,pretrained,args.exp_mode)
    checkpoint_file="%s/deep_speech_final_hidden_size_%s_hidden_layers_%s_rnn_type_%s__augment_%s_pretrained_%s_exp_mode_%s"%(model_dir,args.hidden_size,args.hidden_layers,args.rnn_type,args.augment,pretrained,args.exp_mode)
    args.log_dir="%s/visualize/deep_speech_final_hidden_size_%s_hidden_layers_%s_rnn_type_%s_epochs_%s_augment_%s_pretrained_%s_exp_mode%s"%(args.wdir, args.hidden_size,args.hidden_layers,args.rnn_type,args.epochs,args.augment,pretrained,args.exp_mode)
    loss_results, cer_results, wer_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(
        args.epochs)

    print "log  ", log_file_name
    
    best_wer = None
    best_cer = None
    if args.visdom:
        from visdom import Visdom

        viz = Visdom()
        opts = dict(title=args.id, ylabel='', xlabel='Epoch', legend=['Loss', 'WER', 'CER'])
        viz_window = None
        epochs = torch.arange(1, args.epochs + 1)
    if args.tensorboard:
        try:
            os.makedirs(args.log_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Tensorboard log directory already exists.')
                for file in os.listdir(args.log_dir):
                    file_path = os.path.join(args.log_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception:
                        raise
            else:
                raise
        from tensorboardX import SummaryWriter

        tensorboard_writer = SummaryWriter(args.log_dir)

    try:
        os.makedirs(save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Model Save directory already exists.')
        else:
            raise
    #criterion = CTCLoss()

    avg_loss, start_epoch, start_iter = 0, 0, 0
    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
        model = DeepSpeech.load_model_package(package)
        labels = DeepSpeech.get_labels(model)
        audio_conf = DeepSpeech.get_audio_conf(model)
        parameters = model.parameters()
        optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                    momentum=args.momentum, nesterov=True)
        if not args.finetune:  # Don't want to restart training
            optimizer.load_state_dict(package['optim_dict'])
            if args.cuda:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

            start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training
            start_iter = package.get('iteration', None)
            if start_iter is None:
                start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
                start_iter = 0
            else:
                start_iter += 1
            avg_loss = int(package.get('avg_loss', 0))

            #loss_results, cer_results, wer_results,  best_wer = package['loss_results'], package[
            loss_results, cer_results, wer_results= package['loss_results'], package[
                'cer_results'], package['wer_results'] #, package['best_wer_results']
            if args.visdom and \
                            package[
                                'loss_results'] is not None and start_epoch > 0:  # Add previous scores to visdom graph
                x_axis = epochs[0:start_epoch]
                y_axis = torch.stack(
                    (loss_results[0:start_epoch], wer_results[0:start_epoch], cer_results[0:start_epoch]),
                    dim=1)
                viz_window = viz.line(
                    X=x_axis,
                    Y=y_axis,
                    opts=opts,
                )
            if args.tensorboard and \
                            package[
                                'loss_results'] is not None and start_epoch > 0:  # Previous scores to tensorboard logs
                for i in range(start_epoch):
                    values = {
                        'Avg Train Loss': loss_results[i],
                        'Avg WER': wer_results[i],
                        'Avg CER': cer_results[i]
                    }
                    tensorboard_writer.add_scalars(args.id, values, i + 1)
    else:
        with open(args.labels_path) as label_file:
            data_loaded = json.load(label_file)
            #labels = str(''.join(data_loaded).encode("utf8","ignore"))
            labels = data_loaded
	    #print (" labels ", len(labels))
	  
        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))

        rnn_type = args.rnn_type.lower()
        assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
        model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                           nb_layers=args.hidden_layers,
                           labels=labels,
                           rnn_type=supported_rnns[rnn_type],
                           audio_conf=audio_conf,
                           dec_hidden_size=args.dec_hidden_size,
			   dec_n_layers=args.dec_n_layers,
			   dec_dropout_p=args.dec_dropout, 
                           bidirectional=args.bidirectional)
	if args.pretrained:
	   print('Given pretrained file ', args.pretrained)
	   model.load_from_pretrained_file(args.pretrained)
        parameters = model.parameters()
        optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                    momentum=args.momentum, nesterov=True)
    decoder = GreedyDecoder(labels)
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, augment=args.augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, augment=False)
    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    if not args.no_shuffle and start_epoch != 0:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle()

    if args.cuda:
        #model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()

    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        end = time.time()
        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break
            inputs, targets, target_batch, input_percentages, target_sizes, audio_ids = data
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = Variable(inputs, requires_grad=False)
           # target_sizes = Variable(target_sizes, requires_grad=False)
            targets = Variable(targets, requires_grad=False)
            target_batch = Variable(target_batch, requires_grad=False)
#	    print ("target  ", targets  )
 #           print ("target_batch  ", target_batch  )
  #          print "targets  ", targets, " target_sizes ",target_sizes,"\n"
            if args.cuda:
                inputs = inputs.cuda()
		target_batch = target_batch.cuda() 
            out = model(inputs, target_batch)
            #out = model(inputs)
            out = out.contiguous()  # TxNxH

            seq_length = out.size(0)
            sizes = Variable(input_percentages.mul_(int(seq_length)).int(), requires_grad=False)
#            loss = criterion(out, targets, sizes, target_sizes) :
            #  Test masked cross entropy loss : returns An average loss value masked by the length, and the loss.sum() .
            loss_avg_length, loss= masked_cross_entropy( out, target_batch.contiguous(),target_sizes)
            #print ( " loss_avg_length ", loss_avg_length, " loss  ", loss)
            loss = loss / inputs.size(0)  # average the loss by minibatch
            loss_sum = loss.data.sum()
            inf = float("inf")
            if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss.data[0]

            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)
            # SGD step
            optimizer.step()

            if args.cuda:
                torch.cuda.synchronize()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.silent:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time,
                    data_time=data_time, loss=losses))
            if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0:
                file_path = '%s/deepspeech_checkpoint_epoch_%d_iter_%d.pth.tar' % (save_folder, epoch + 1, i + 1)
                print("Saving checkpoint model to %s" % file_path)
                torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                loss_results=loss_results,
                                                wer_results=wer_results, cer_results=cer_results, best_wer_results=best_wer, avg_loss=avg_loss),
                           file_path)
            del loss
            del out
        print (" avg_loss  ", avg_loss  , " train_sampler  ", len(train_sampler), " avg_loss /= len(train_sampler) ", (avg_loss / len(train_sampler)))
        avg_loss /= len(train_sampler)

        print('Training Summary Epoch: [{0}]\t'
              'Average Loss {loss:.3f}\t'.format(
            epoch + 1, loss=avg_loss))

        start_iter = 0  # Reset start iteration for next epoch
        total_cer, total_wer = 0, 0
        model.eval()
	print (" Eval  \n")
        for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs, targets, target_batch,  input_percentages, target_sizes,audio_ids = data
            inputs = Variable(inputs, volatile=True)
	    target_batch = Variable(target_batch, volatile=True)
            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size
	   
#	    print ("target  ", targets  )
#	    print ("target_batch  ", target_batch  )
#	    print "targets  ", targets, " target_sizes ",target_sizes,"\n"
            if args.cuda:
                inputs = inputs.cuda()
		target_batch = target_batch.cuda()
            out = model(inputs, target_batch)
#            print (" OUT  ", out)
            out = out.transpose(0, 1)  # TxNxH
            seq_length = out.size(0)
            sizes = input_percentages.mul_(int(seq_length)).int()

            decoded_output, _ = decoder.decode(out.data, sizes)
            target_strings = decoder.convert_to_strings(split_targets)
            wer, cer = 0, 0
            for x in range(len(target_strings)):
                transcript, reference = decoded_output[x][0], target_strings[x][0]
                wer += decoder.wer(transcript, reference) / float(len(reference.split()))
                cer += decoder.cer(transcript, reference) / float(len(reference))
            total_cer += cer
            total_wer += wer

            if args.cuda:
                torch.cuda.synchronize()
            del out
        wer = total_wer / len(test_loader.dataset)
        cer = total_cer / len(test_loader.dataset)
        wer *= 100
        cer *= 100
        loss_results[epoch] = avg_loss
        wer_results[epoch] = wer
        cer_results[epoch] = cer
        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(
            epoch + 1, wer=wer, cer=cer))
        log_file=open(log_file_name,'a')
	log_file.write(' Validation Summary Epoch: '+str(epoch + 1)+' Average WER '+ str(wer)+'  Average CER '+str(cer)+'\n')
        if args.visdom:
            x_axis = epochs[0:epoch + 1]
            y_axis = torch.stack((loss_results[0:epoch + 1], wer_results[0:epoch + 1], cer_results[0:epoch + 1]), dim=1)
            if viz_window is None:
                viz_window = viz.line(
                    X=x_axis,
                    Y=y_axis,
                    opts=opts,
                )
            else:
                viz.line(
                    X=x_axis.unsqueeze(0).expand(y_axis.size(1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                    Y=y_axis,
                    win=viz_window,
                    update='replace',
                )
        if args.tensorboard:
            values = {
                'Avg Train Loss': avg_loss,
                'Avg WER': wer,
                'Avg CER': cer
            }
            tensorboard_writer.add_scalars(args.id, values, epoch + 1)
            if args.log_params:
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    tensorboard_writer.add_histogram(tag, to_np(value), epoch + 1)
                    tensorboard_writer.add_histogram(tag + '/grad', to_np(value.grad), epoch + 1)
        if args.checkpoint and epoch % 5  == 0 :
            file_path = '%s_epoch_%d_exp_mode_%s.pth.tar' % (checkpoint_file, epoch + 1, args.exp_mode)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                            wer_results=wer_results, cer_results=cer_results,  best_wer_results=best_wer),
                       file_path)
        # anneal lr
        optim_state = optimizer.state_dict()
        optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / args.learning_anneal
        optimizer.load_state_dict(optim_state)
        print('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

        if best_wer is None or best_wer > wer:
            model_path="%s/deep_speech_final_hidden_size_%s_hidden_layers_%s_rnn_type_%s_epochs_%s_augment_%s_pretrained_%s_exp_mode_%s_best_wer_%s.pth.tar"%(model_dir,args.hidden_size,args.hidden_layers,args.rnn_type,args.epochs,args.augment,pretrained,args.exp_mode,str(best_wer))
            print("Found better validated model, saving to %s" % model_path)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                            wer_results=wer_results, cer_results=cer_results, best_wer_results=best_wer)
                       , model_path)
            best_wer = wer
	    best_cer = cer
            
	    log_file.write(' Found better validated model epoch: '+str(epoch + 1)+' Average WER '+ str(wer)+'  Average CER '+str(cer)+'\n')
	    log_file
