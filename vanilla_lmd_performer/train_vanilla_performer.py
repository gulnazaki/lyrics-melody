import deepspeed
from performer_pytorch import PerformerEncDec
import argparse
import random
import pandas as pd
import json
from allennlp.training.metrics import BLEU
from itertools import cycle
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split


def get_arguments():
    parser=argparse.ArgumentParser(description='Train Vanilla Performer on Lakh Midi Dataset Instruments-Lyrics-Vocal Melody')

    parser.add_argument('--dataset-file', '-df', type=str, required=True,
                        help='Dataset parquet file')

    parser.add_argument('--vocabulary-prefix', '-v', type=str, default='',
                        help='Prefix of the vocab files: <pref>_instrumental.vocab, <prf>_vocal.vocab')

    parser.add_argument('--save-dir', '-sd', type=str, required=True,
                        help='Directory to save checkpoints, states, event logs')
    
    parser.add_argument('--monophonic', '-m', default=False, action='store_true',
                        help='Use monophonic instead of full instrumental input')

    parser.add_argument('--max-instrumental-sequence-length', '-maxi', type=int, default=-1,
                        help='If provided it will truncate samples with longer instrumental sequences')
    
    parser.add_argument('--max-vocal-sequence-length', '-maxv', type=int, default=-1,
                        help='If provided it will truncate samples with longer vocal melody sequences')
    
    parser.add_argument('--train-split', '-ts', type=float, default=0.9,
                        help='Percentage of the dataset to use for training')

    parser.add_argument('--epochs', '-e', type=int, default=20,
                        help='Number of epochs')
    
    parser.add_argument('--validate-every', '-ve', type=int, default=200,
                        help='Validate every n batches')
    
    parser.add_argument('--generate-every', '-ge', type=int, default=400,
                        help='Generate every n batches')

    parser.add_argument('--print-training-loss-every', '-ptle', type=int, default=20,
                        help='It will average training loss and print it every n steps')

    parser.add_argument('--validate-size', '-vs', type=int, default=40,
                        help='Will calculate average of validation loss for n batches')

    parser.add_argument('--validate-batch-size', '-vss', type=int, default=1,
                        help='Batch size for validation dataset')

    parser.add_argument('--checkpoints-per-epoch', '-cpp', type=int, default=3,
                        help='How many checkpoints to keep per epoch')
    
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank passed from distributed launcher')
    
    parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()


class MidiDataset(Dataset):
    def __init__(self, dataset_file, monophonic, vocabulary_prefix, max_instrumental_length, max_vocal_length):
        super().__init__()
        instrumental_type = 'monophonic' if monophonic else 'instrumental'
        with open('{}instrumental.vocab'.format(vocabulary_prefix), 'r') as f, \
            open('{}vocal.vocab'.format(vocabulary_prefix), 'r') as g: 
            self.instrumental_vocab = {w : l for l, w in enumerate(f.read().splitlines())}
            self.reverse_instrumental_vocab = {l: w for w, l in self.instrumental_vocab.items()}
            self.vocal_vocab = {w : l for l, w in enumerate(g.read().splitlines())}
            self.reverse_vocal_vocab = {l: w for w, l in self.vocal_vocab.items()}
            
        df = pd.read_parquet(dataset_file)

        self.files = list(df['file'])
        self.instrumental = [self.encode(json.loads(f), seq_type='instrumental', max_length=max_instrumental_length) for f in df[instrumental_type]]
        self.vocals = [self.encode(json.loads(v), seq_type='vocals', max_length=max_vocal_length) for v in df['vocal']]

        self.max_instrumental_length = max([len(f) for f in self.instrumental])
        self.max_vocal_length = max([len(f) for f in self.vocals])


    def __getitem__(self, index):
        return (self.instrumental[index], self.vocals[index]), self.files[index]

    def __len__(self):
        return len(self.files)

    def truncate(self, sequence, max_length):
        if max_length >= 0:
            return sequence[:max_length]
        return sequence

    def encode(self, event_sequence, seq_type, max_length=-1):
        if seq_type == 'instrumental':
            return torch.tensor([self.instrumental_vocab[e] for e in self.truncate(event_sequence, max_length - 1)] + [self.instrumental_vocab['<eos>']])
        else:
            return torch.tensor([self.vocal_vocab['<bos>']] + [self.vocal_vocab[e] for e in self.truncate(event_sequence, max_length - 2)] + [self.vocal_vocab['<eos>']])

    def decode(self, event_sequence, seq_type, mask=None):
        size = len(event_sequence)
        if mask is not None:
            mask = mask.tolist()
            true_size = len([v for v in mask if v])
        else:
            true_size = size
        if seq_type == 'instrumental':
            return [self.reverse_instrumental_vocab[i.item()] for i in event_sequence[:true_size]]
        else:
            return [self.reverse_vocal_vocab[o.item()] for o in event_sequence[:true_size]]


def collate_fn_zero_pad(batch):
    data, files = zip(*batch)
    instrumental, vocals = zip(*data)
    batch_size = len(files)

    if batch_size == 1:
        instrumental = instrumental[0].view(1, -1)
        vocals = vocals[0].view(1, -1)
        instrumental_masks = torch.ones_like(instrumental).bool()
        vocal_masks = torch.ones_like(vocals).bool()
        return (instrumental.long(), instrumental_masks), (vocals.long(), vocal_masks), files[0]

    instrumental_lengths = [seq.size(0) for seq in instrumental]
    instrumental_max_length = max(instrumental_lengths)
    instrumental_masks = torch.arange(instrumental_max_length).view(1, -1).expand(batch_size, -1) < torch.tensor(instrumental_lengths).view(-1, 1)
    padded_instrumental = torch.zeros(batch_size, instrumental_max_length)
    for i, l in enumerate(instrumental_lengths):
        padded_instrumental[i, :l] = instrumental[i]

    vocal_lengths = [seq.size(0) for seq in vocals]
    vocal_max_length = max(vocal_lengths)
    vocal_masks = torch.arange(vocal_max_length).view(1, -1).expand(batch_size, -1) < torch.tensor(vocal_lengths).view(-1, 1)
    padded_vocals = torch.zeros(batch_size, vocal_max_length)
    for i, l in enumerate(vocal_lengths):
        padded_vocals[i, :l] = vocals[i]

    return (padded_instrumental.long(), instrumental_masks), (padded_vocals.long(), vocal_masks), files


def valid_structure_metric(sequence, vocab):
    def get_valids_for_next(e, note_was_on):
        if e == waits[-1]:
            valid_events = waits + offs + boundaries + phonemes
        elif e in waits:
            valid_events = offs + boundaries + phonemes
        elif e in ons:
            note_was_on = True
            valid_events = waits
        elif e in offs:
            note_was_on = False
            valid_events = waits + boundaries + phonemes
        elif e in boundaries:
            if e == boundaries[-1]:
                valid_events = boundaries[:-1] + phonemes
            else:
                valid_events = phonemes
        else:
            valid_events = ons
        return valid_events, note_was_on

    sequence = sequence.tolist()
    waits = [i for e, i in vocab.items() if e[:2] == 'W_']
    ons = [i for e, i in vocab.items() if e[:3] == 'ON_']
    offs = [vocab['_OFF_']]
    boundaries = [vocab[e] for e in ['N_DL', 'N_L', 'N_W', '_C_']]
    phonemes = [i for e, i in vocab.items() if not '_' in e or e == '_R_']
    
    valid_count = 0
    valid_events = waits + phonemes + boundaries
    note_was_on = False
    for e in sequence:
        if e in valid_events and \
        (e not in ons or note_was_on == False) and \
        (e not in offs or note_was_on == True):
            valid_count += 1
        valid_events, note_was_on = get_valids_for_next(e, note_was_on)

    size = len(sequence) - 1 if sequence[-1] == 2 else len(sequence)
    if size == 0:
        return 0
    else:
        return valid_count / size


if __name__ == '__main__':
    args = get_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MidiDataset(dataset_file=args.dataset_file,
                          monophonic=args.monophonic,
                          vocabulary_prefix=args.vocabulary_prefix,
                          max_instrumental_length=args.max_instrumental_sequence_length,
                          max_vocal_length=args.max_vocal_sequence_length)

    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    torch.manual_seed(0)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_log_dir = os.path.join(args.save_dir, 'train')
    val_log_dir = os.path.join(args.save_dir, 'val')
    Path(train_log_dir).mkdir(parents=True, exist_ok=True)
    Path(val_log_dir).mkdir(parents=True, exist_ok=True)
    writer_train = SummaryWriter(log_dir=train_log_dir)
    writer_val = SummaryWriter(log_dir=val_log_dir)
    
    bleu = BLEU()

    model = PerformerEncDec(
        dim = 512,
        enc_heads = 8,
        dec_heads = 8,
        enc_depth = 6,
        dec_depth = 6,
        enc_ff_chunks = 10,
        dec_ff_chunks = 10,
        enc_num_tokens = len(dataset.instrumental_vocab),
        dec_num_tokens = len(dataset.vocal_vocab),
        enc_max_seq_len = dataset.max_instrumental_length,
        dec_max_seq_len = dataset.max_vocal_length,
        enc_emb_dropout = 0.1,
        dec_emb_dropout = 0.1,
        enc_ff_dropout = 0.1,
        dec_ff_dropout = 0.1,
        enc_attn_dropout = 0.1,
        dec_attn_dropout = 0.1,
        enc_tie_embed = True,
        dec_tie_embed = True,
        enc_reversible = True,
        dec_reversible = True
    ).to(device)

    model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters(),  training_data=train_dataset, collate_fn=collate_fn_zero_pad)
    device = model_engine.local_rank

    torch.manual_seed(torch.initial_seed())
    val_loader_ = DataLoader(val_dataset, batch_size=args.validate_batch_size, shuffle=True, collate_fn=collate_fn_zero_pad)
    val_loader = cycle(val_loader_)

    num_batches = (len(train_dataset) + trainloader.batch_size - 1) // trainloader.batch_size

    save_every = num_batches // args.checkpoints_per_epoch
    save_at = 0
    saving_steps = []
    for _ in range(args.checkpoints_per_epoch - 1):
        save_at += save_every
        saving_steps.append(save_at)
    saving_steps.append(num_batches - 1)

    print("\n", "Dataset maximum sequence lengths - Instrumental: {}, Vocal: {}".format(dataset.max_instrumental_length, dataset.max_vocal_length), "\n")
    print("\n", "Train Dataset - size: {}, batches: {}".format(len(train_dataset), num_batches), "\n")
    print("\n", "Validate Dataset - size: {}, batches: {}".format(len(val_dataset), len(val_loader_)), "\n")

    checkpoint_name, client_state = model_engine.load_checkpoint(args.save_dir, load_module_strict=False)
    # checkpoint_name = None

    if checkpoint_name is not None:
        print("\nLoaded checkpoint: {}\n".format(checkpoint_name))        
        i = client_state['i']
        i += 1
        epoch, step = divmod(i, num_batches)
        print("Epoch: {}, step: {}, i: {}".format(epoch, step, i))
        if step == 0:
            print("Starting next epoch...")
            rng = torch.get_rng_state()
            trainloader = iter(trainloader)
        else:
            rng = torch.load(os.path.join(args.save_dir, 'rng_state.pt'))
            torch.set_rng_state(rng)
            trainloader = iter(trainloader)
            print("Advancing dataloader...")
            for _ in range(step):
                next(trainloader)
    else:
        print("\nNo checkpoint found, training from scratch\n")
        i = 0
        step = 0
        epoch = 0
        rng = torch.get_rng_state()
        trainloader = iter(trainloader)


    for e in range(args.epochs - epoch):
        running_loss = 0
        running_loss_steps = 0
        print("EPOCH: {}".format(e + epoch))
        while True:
            try:
                data = next(trainloader)
            except StopIteration:
                step = 0
                rng = torch.get_rng_state()
                trainloader = iter(trainloader)
                break

            model_engine.train()
            (instrumental, instrumental_mask), (vocals, vocals_mask), _ = data
            loss = model_engine(instrumental.to(device),
                                vocals.to(device),
                                enc_mask=instrumental_mask.to(device),
                                dec_mask=vocals_mask.to(device))
            model_engine.backward(loss)
            model_engine.step()
            
            running_loss += loss.item()
            running_loss_steps += 1
            if running_loss_steps == args.print_training_loss_every or step == 0:
                avg_loss = running_loss / running_loss_steps
                print("training loss: {}".format(avg_loss))
                writer_train.add_scalar("Loss", avg_loss, i)
                writer_train.flush()
                running_loss = 0
                running_loss_steps = 0

            if step % args.validate_every == 0:
                model_engine.eval()
                with torch.no_grad():
                    running_eval_loss = 0
                    for _ in range(args.validate_size):
                        (instrumental, instrumental_mask), (vocals, vocals_mask), _ = next(val_loader)
                        loss = model_engine(instrumental.to(device),
                                            vocals.to(device),
                                            enc_mask=instrumental_mask.to(device),
                                            dec_mask=vocals_mask.to(device))
                        running_eval_loss += loss.item()
                    avg_eval_loss = running_eval_loss / args.validate_size
                    print('\n', f'validation loss: {avg_eval_loss}', '\n')
                    writer_val.add_scalar("Loss", avg_eval_loss, i)
                    writer_val.flush()
                    running_eval_loss = 0

            if step % args.generate_every == 0:
                (instrumental, instrumental_mask), (expected_vocals, expected_vocals_mask), file = next(val_loader)
                decoded_expected_vocals = dataset.decode(expected_vocals[0][1:], seq_type='vocals', mask=expected_vocals_mask[0][1:])

                instrumental = instrumental[0].view(1, -1)
                instrumental_mask = instrumental_mask[0].view(1, -1)
                
                # <bos> token
                vocals_start = torch.ones(1,1).long()

                vocals = model_engine.module.generate(instrumental.to(device),
                                                      vocals_start.to(device),
                                                      seq_len=dataset.max_vocal_length//8,
                                                      enc_mask=instrumental_mask.to(device),
                                                      eos_token=2)
                decoded_vocals = dataset.decode(vocals[0], seq_type='vocals')

                with open(os.path.join(args.save_dir, 'outputs.txt'), 'a') as f:
                    f.write("{}:\n\n{}\n----------------\n{}\n----------------\n\n"\
                                    .format(file, decoded_expected_vocals, decoded_vocals))
                
                bleu(vocals.to(device), expected_vocals[:, 1:].to(device))
                b = bleu.get_metric(reset=True)['BLEU']
                print("BLEU metric: {}".format(b))

                vsm = valid_structure_metric(vocals[0], dataset.vocal_vocab)
                print("Valid Structure Metric: {}".format(vsm))
                # expected_vsm = valid_structure_metric(expected_vocals[0][1:], dataset.vocal_vocab)
                # print("Expected Valid Structure Metric: {} (for control)".format(expected_vsm))
                writer_val.add_scalar("BLEU", b, i)
                writer_val.add_scalar("VSM", vsm, i)
                writer_val.flush()

            if step in saving_steps:
                loss_to_ckpt = avg_eval_loss if avg_eval_loss is not None else loss.item()
                ckpt_id = "{}-{}-{}".format(e + epoch, i, loss_to_ckpt)
                model_engine.save_checkpoint(args.save_dir, tag=ckpt_id, client_state = {'i': i, 'step': step, 'epoch': e + epoch})
                torch.save(rng, os.path.join(args.save_dir, 'rng_state.pt'))
                torch.save(model_engine.module.state_dict(), os.path.join(args.save_dir, 'model.pt'))

            i += 1
            step += 1
            