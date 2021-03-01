import deepspeed
from performer_pytorch import PerformerLM, AutoregressiveWrapper
import argparse
import random
import pandas as pd
import re
from itertools import cycle
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer
from statistics import mean


def get_arguments():
    parser=argparse.ArgumentParser(description='Finetune LM on lyrics')

    parser.add_argument('--dataset-file', '-df', type=str, required=True,
                        help='Dataset parquet file')

    parser.add_argument('--pretrained-model', '-pm', type=str,
                        help='Pretrained huggingface model to load')

    parser.add_argument('--tokenizer', '-tok', type=str,
                        help='Hugginface tokenizer to use')

    parser.add_argument('--max-seq-len', '-msl', type=int, default=1024,
                        help='Max sequence length')

    parser.add_argument('--save-dir', '-sd', type=str, required=True,
                        help='Directory to save checkpoints, states, event logs')
    
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


class LyricsDataset(Dataset):
    def __init__(self, dataset_file, tokenizer, max_length=1024):
        super().__init__()
        
        df = pd.read_parquet(dataset_file)
        self.lyrics = list(df['Lyrics'])

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)

        self.max_length = max_length
        self.max_seq_len = 0
        self.mean_seq_len = 0

    def __getitem__(self, index):
        return torch.tensor([self.tokenizer.bos_token_id] + self.tokenizer.encode(self.lyrics[index], max_length=self.max_length - 2, truncation=True) + [self.tokenizer.eos_token_id])

    def __len__(self):
        return len(self.lyrics)

    def batch_encode(self, sequences):
        return self.tokenizer.batch_encode_plus(sequences)['input_ids']

    def batch_decode(self, sequences, masks=None):
        if masks is None:
            return self.tokenizer.batch_decode(sequences)

        batch = []
        for sequence, mask in zip(sequences, masks):
            size = len(sequence)
            mask = mask.tolist()
            true_size = len([v for v in mask if v])
            batch.append(self.tokenizer.decode(sequence[:true_size]))
        return batch


def collate_fn_zero_pad(batch):
    batch_size = len(batch)

    if batch_size == 1:
        data = batch[0].view(1, -1)
        masks = torch.ones_like(data).bool()
        return (data, masks)

    lengths = [seq.size(0) for seq in batch]
    max_length = max(lengths)
    masks = torch.arange(max_length).view(1, -1).expand(batch_size, -1) < torch.tensor(lengths).view(-1, 1)
    padded_data = torch.zeros(batch_size, max_length)
    for i, l in enumerate(lengths):
        padded_data[i, :l] = batch[i]

    return (padded_data.long(), masks)


if __name__ == '__main__':
    args = get_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = LyricsDataset(dataset_file=args.dataset_file,
                            tokenizer=args.tokenizer,
                            max_length=args.max_seq_len)

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
    
    model = PerformerLM(
            dim = 768,
            heads = 12,
            depth = 6,
            num_tokens = len(dataset.tokenizer),
            max_seq_len = args.max_seq_len,
            emb_dropout = 0.1,
            ff_dropout = 0.1,
            attn_dropout = 0.1,
            tie_embed = True,
            reversible = True,
            causal = True
    )

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model), strict=False)
        print("Loaded pretrained model: {}".format(args.pretrained_model))

    model = AutoregressiveWrapper(model).to(device)

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

    print("\n", "Dataset maximum sequence length: {} Dataset mean sequence length: {}".format(dataset.max_seq_len, dataset.mean_seq_len, "\n"))
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
            lyrics, mask = data
            loss = model_engine(lyrics.to(device), mask=mask.to(device), return_loss=True)
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
                        lyrics, mask = next(val_loader)
                        loss = model_engine(lyrics.to(device), mask=mask.to(device), return_loss=True)
                        running_eval_loss += loss.item()
                    avg_eval_loss = running_eval_loss / args.validate_size
                    print('\n', f'validation loss: {avg_eval_loss}', '\n')
                    writer_val.add_scalar("Loss", avg_eval_loss, i)
                    writer_val.flush()
                    running_eval_loss = 0

            if step % args.generate_every == 0:
                # <bos> token
                initial = torch.full((1,1), dataset.tokenizer.bos_token_id).long()

                outs = [model_engine.module.generate(initial.to(device), seq_len=args.max_seq_len - 2, eos_token=dataset.tokenizer.eos_token_id)[0] for _ in range(4)]
                decoded_outs = '\n--------\n'.join(dataset.batch_decode(outs))
                # print(decoded_outs)

                with open(os.path.join(args.save_dir, 'outputs.txt'), 'a') as f:
                    f.write(decoded_outs + '\n--------\n')

            if step in saving_steps:
                loss_to_ckpt = avg_eval_loss if avg_eval_loss is not None else loss.item()
                ckpt_id = "{}-{}-{}".format(e + epoch, i, loss_to_ckpt)
                model_engine.save_checkpoint(args.save_dir, tag=ckpt_id, client_state = {'i': i, 'step': step, 'epoch': e + epoch})
                torch.save(rng, os.path.join(args.save_dir, 'rng_state.pt'))

            i += 1
            step += 1
