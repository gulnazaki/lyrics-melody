import deepspeed
from performer_pytorch import PerformerEncDec
import argparse
import random
import pandas as pd
import json
from tqdm import tqdm
from allennlp.training.metrics import BLEU
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split


def get_arguments():
    parser=argparse.ArgumentParser(description='Lakh Midi Dataset Instruments-Vocals')

    parser.add_argument('--dataset-file', '-df', type=str, required=True,
                        help='Dataset parquet file')

    parser.add_argument('--vocabulary-prefix', '-v', type=str, default='',
                        help='Prefix of the vocab files: <pref>_instrumental.vocab, <prf>_vocal.vocab')
    
    parser.add_argument('--monophonic', '-m', default=False, action='store_true',
                        help='Use monophonic instead of full instrumental input')

    parser.add_argument('--max-input-sequence-length', '-maxi', type=int, default=-1,
                        help='If provided it will skip samples with longer input sequences')
    
    parser.add_argument('--max-output-sequence-length', '-maxo', type=int, default=-1,
                        help='If provided it will skip samples with longer output sequences')
    
    parser.add_argument('--train-split', '-ts', type=float, default=0.9,
                        help='Percentage of the dataset to use for training')

    parser.add_argument('--epochs', '-e', type=int, default=20,
                        help='Number of epochs')
    
    parser.add_argument('--validate-every', '-ve', type=int, default=200,
                        help='Validate every n batches')
    
    parser.add_argument('--generate-every', '-ge', type=int, default=500,
                        help='Generate every n batches')
    
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank passed from distributed launcher')
    
    parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()


class MidiDataset(Dataset):
    def __init__(self, dataset_file, monophonic, vocabulary_prefix, max_input_length, max_output_length):
        super().__init__()
        input_type = 'monophonic' if monophonic else 'instrumental'
        with open('{}instrumental.vocab'.format(vocabulary_prefix), 'r') as f, \
            open('{}vocal.vocab'.format(vocabulary_prefix), 'r') as g: 
            self.input_vocab = {w : l for l, w in enumerate(f.read().splitlines())}
            self.reverse_input_vocab = {l: w for w, l in self.input_vocab.items()}
            self.output_vocab = {w : l for l, w in enumerate(g.read().splitlines())}
            self.reverse_output_vocab = {l: w for w, l in self.output_vocab.items()}
            
        df = pd.read_parquet(dataset_file, columns=['vocal', input_type])
        
        inp = [self.encode(json.loads(f) + ['<eos>'], is_input=True) for f in df[input_type]]
        out = [self.encode(['<bos>'] + json.loads(f) + ['<eos>'], is_input=False) for f in df['vocal']]

        if max_input_length < 0 and max_output_length < 0:
            self.input = inp
            self.output = out
        else:
            self.input = []
            self.output = []
            for idx in range(len(inp)):
                input_sample = inp[idx]
                output_sample = out[idx]
                if (max_input_length >= 0 and len(input_sample) > max_input_length) or \
                   (max_output_length >= 0 and len(output_sample) > max_output_length):
                   continue
                else:
                    self.input.append(input_sample)
                    self.output.append(output_sample)

        self.max_input_length = max([len(f) for f in self.input])
        self.max_output_length = max([len(f) for f in self.output])


    def __getitem__(self, index):
        return (self.input[index], self.output[index])

    def __len__(self):
        return len(self.input)

    def encode(self, event_sequence, is_input):
        if is_input:
            return torch.tensor([self.input_vocab[i] for i in event_sequence])
        else:
            return torch.tensor([self.output_vocab[i] for i in event_sequence])

    def decode(self, event_sequence, is_input, mask=None):
        size = len(event_sequence)
        if mask is not None:
            mask = mask.tolist()
            true_size = len([v for v in mask if v])
        else:
            true_size = size
        if is_input:
            return ",".join([self.reverse_input_vocab[i.item()] for i in event_sequence[:true_size]])
        else:
            return ",".join([self.reverse_output_vocab[o.item()] for o in event_sequence[:true_size]])

def collate_fn_zero_pad(batch):
    inputs, outputs = zip(*batch)
    batch_size = len(inputs)

    if batch_size == 1:
        inputs = inputs[0].view(1, -1)
        outputs = outputs[0].view(1, -1)
        input_masks = torch.ones_like(inputs).bool()
        output_masks = torch.ones_like(outputs).bool()
        return (inputs.long(), input_masks), (outputs.long(), output_masks)

    input_lengths = [seq.size(0) for seq in inputs]
    input_max_length = max(input_lengths)
    input_masks = torch.arange(input_max_length).view(1, -1).expand(batch_size, -1) < torch.tensor(input_lengths).view(-1, 1)
    padded_inputs = torch.zeros(batch_size, input_max_length)
    for i, l in enumerate(input_lengths):
        padded_inputs[i, :l] = inputs[i]

    output_lengths = [seq.size(0) for seq in outputs]
    output_max_length = max(output_lengths)
    output_masks = torch.arange(output_max_length).view(1, -1).expand(batch_size, -1) < torch.tensor(output_lengths).view(-1, 1)
    padded_outputs = torch.zeros(batch_size, output_max_length)
    for i, l in enumerate(output_lengths):
        padded_outputs[i, :l] = outputs[i]

    return (padded_inputs.long(), input_masks), (padded_outputs.long(), output_masks)


if __name__ == '__main__':
    args = get_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MidiDataset(dataset_file=args.dataset_file,
                          monophonic=args.monophonic,
                          vocabulary_prefix=args.vocabulary_prefix,
                          max_input_length=args.max_input_sequence_length,
                          max_output_length=args.max_output_sequence_length)

    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    torch.manual_seed(0)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    writer_train = SummaryWriter(log_dir='drive/MyDrive/lmd_transformer/small/train')
    writer_val = SummaryWriter(log_dir='drive/MyDrive/lmd_transformer/small/val')
    bleu = BLEU()

    model = PerformerEncDec(
        dim = 512,
        enc_heads = 8,
        dec_heads = 8,
        enc_depth = 6,
        dec_depth = 6,
        enc_ff_chunks = 10,
        dec_ff_chunks = 10,
        enc_num_tokens = len(dataset.input_vocab),
        dec_num_tokens = len(dataset.output_vocab),
        enc_max_seq_len = dataset.max_input_length,
        dec_max_seq_len = dataset.max_output_length,
        enc_emb_dropout = 0.1,
        dec_emb_dropout = 0.1,
        enc_ff_dropout = 0.1,
        dec_ff_dropout = 0.1,
        enc_attn_dropout = 0.1,
        dec_attn_dropout = 0.1,
        enc_reversible = True,
        dec_reversible = True
    ).to(device)

    model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters(),  training_data=train_dataset, collate_fn=collate_fn_zero_pad)
    device = model_engine.local_rank

    val_loader_ = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_zero_pad)
    val_loader = cycle(val_loader_)


    i = None
    trainloader = iter(trainloader)
    checkpoint_name, client_state = model_engine.load_checkpoint('drive/MyDrive/lmd_transformer/small/')

    print("\n", "Dataset maximum sequence lengths - Input: {}, Output: {}".format(dataset.max_input_length, dataset.max_output_length), "\n")
    print("\n", "Train Dataset - size: {}, batches: {}".format(len(train_dataset), len(trainloader.dataloader)), "\n")
    print("\n", "Validate Dataset - size: {}, batches: {}".format(len(val_dataset), len(val_loader_)), "\n")

    if checkpoint_name is not None:
        print("\nLoaded checkpoint: {}\n".format(checkpoint_name))        
        i = client_state['i']
        i += 1
        epoch, step = divmod(i, len(trainloader.dataloader))
        print("Epoch: {}, step: {}, i: {}".format(epoch, step, i))
        print("Advancing dataloader...")
        for _ in tqdm(range(step)):
            next(trainloader)
    else:
        print("\nNo checkpoint found, training from scratch\n")

    if i is None:
        i = 0
        step = 0
        epoch = 0

    for e in range(args.epochs - epoch):
        running_loss = 0
        print("EPOCH: {}".format(e + epoch))
        while True:
            try:
                data = next(trainloader)
            except StopIteration:
                ckpt_id = "end_of_epoch_{}-{}-{}".format(e + epoch, i - 1, loss.item())
                model_engine.save_checkpoint('drive/MyDrive/lmd_transformer/small/', tag=ckpt_id, client_state = {'i' : i - 1})
                step = 0
                trainloader = iter(trainloader)
                break

            model_engine.train()
            (inp, inp_mask), (out, out_mask) = data
            loss = model_engine(inp.to(device), out.to(device), enc_mask=inp_mask.to(device), dec_mask=out_mask.to(device), return_loss=True)
            model_engine.backward(loss)
            model_engine.step()
            running_loss += loss.item()
            if step % 20 == 0:
                avg_loss = running_loss / 20 if step > 0 else running_loss
                print("training loss: {}".format(avg_loss))
                writer_train.add_scalar("Loss/train", avg_loss, i)
                writer_train.flush()
                running_loss = 0

            if step % args.validate_every == 0:
                model_engine.eval()
                with torch.no_grad():
                    running_eval_loss = 0
                    for _ in range(40):
                        (inp, inp_mask), (out, out_mask) = next(val_loader)
                        loss = model_engine(inp.to(device), out.to(device), return_loss=True, enc_mask=inp_mask.to(device), dec_mask=out_mask.to(device))
                        running_eval_loss += loss.item()
                    print('\n', f'validation loss: {running_eval_loss / 40}', '\n')
                    writer_val.add_scalar("Loss/evaluate", running_eval_loss / 40, i)
                    writer_val.flush()
                    running_eval_loss = 0

            if step % args.generate_every == 0:
                (inp, inp_mask), (expected_out, expected_out_mask) = next(val_loader)
                print(dataset.decode(inp[0], is_input=True, mask=inp_mask[0]))
                print(dataset.decode(expected_out[0][1:], is_input=False, mask=expected_out_mask[0][1:]))

                inp = inp[0].view(1, -1)
                inp_mask = inp_mask[0].view(1, -1)
                # <bos> token
                initial = torch.ones(1,1).long()

                out = model_engine.module.generate(inp.to(device), initial.to(device), enc_mask=inp_mask.to(device), seq_len=len(expected_out[0]) - 2, eos_token=2)
                print(dataset.decode(out[0], is_input=False))
                
                bleu(out.to(device), expected_out[:, 1:].to(device))
                b = bleu.get_metric(reset=True)['BLEU']
                print("BLEU metric: {}".format(b))
                writer_val.add_scalar("BLEU", b, i)
                writer_val.flush()

            if step == 700 :
                ckpt_id = "midepoch_{}-{}-{}".format(e + epoch, i, loss.item())
                model_engine.save_checkpoint('drive/MyDrive/lmd_transformer/small/', tag=ckpt_id, client_state = {'i' : i})
            
            i += 1
            step += 1
