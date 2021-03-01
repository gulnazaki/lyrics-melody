import re
import torch
from torch import nn
from performer_pytorch.performer_pytorch import PerformerLM
from performer_pytorch.autoregressive_wrapper import AutoregressiveWrapper

ENC_PREFIX = 'enc_'
LM_PREFIX = 'lm_'
DEC_PREFIX = 'dec_'

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return bool(re.match(f'^{prefix}', str))

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(lambda x: string_begins_with(prefix, x), d)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: string_begins_with(prefix, x), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def extract_enc_lm_dec_kwargs(kwargs):
    enc_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(ENC_PREFIX, kwargs)
    lm_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(LM_PREFIX, kwargs)
    dec_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(DEC_PREFIX, kwargs)
    return enc_kwargs, lm_kwargs, dec_kwargs, kwargs

def extract_and_set_enc_lm_dec_kwargs(kwargs):
    enc_kwargs, lm_kwargs, dec_kwargs, kwargs = extract_enc_lm_dec_kwargs(kwargs)
    if 'mask' in enc_kwargs:
        dec_kwargs.setdefault('context_mask', enc_kwargs['mask'])
    if 'mask' in lm_kwargs:
        dec_kwargs.setdefault('second_context_mask', lm_kwargs['mask'][:, :-1])
    return enc_kwargs, lm_kwargs, dec_kwargs, kwargs

class DecoupledPerformer(nn.Module):
    def __init__(
        self,
        dim,
        tie_token_embeds = False,
        no_projection = False,
        pretrained_lm = "",
        **kwargs
    ):
        super().__init__()
        enc_kwargs, lm_kwargs, dec_kwargs, _ = extract_enc_lm_dec_kwargs(kwargs)
        
        assert 'dim' not in enc_kwargs and 'dim' not in dec_kwargs and 'dim' not in lm_kwargs

        enc_kwargs['dim'] = lm_kwargs['dim'] = dec_kwargs['dim'] = dim
        enc_kwargs['no_projection'] = lm_kwargs['no_projection'] = dec_kwargs['no_projection'] = no_projection

        lm_kwargs['causal'] = True
        # cross attention has to be set explicitly
        if not 'cross_attend' in lm_kwargs:
            lm_kwargs['cross_attend'] = False
        
        self.lm_cross_attending = lm_kwargs['cross_attend']

        dec_kwargs['causal'] = True
        dec_kwargs['cross_attend'] = True
        dec_kwargs['second_cross_attend'] = True

        enc = PerformerLM(**enc_kwargs)
        lm = PerformerLM(**lm_kwargs)
        dec = PerformerLM(**dec_kwargs)

        if tie_token_embeds:
            enc.token_emb = lm.token_emb = dec.token_emb

        self.enc = enc
        if pretrained_lm:
            pretrained = torch.load(pretrained_lm)
            from collections import OrderedDict
            new_pretrained = OrderedDict()
            if lm_kwargs['reversible']:
                if lm_kwargs['cross_attend']:
                    for k, v in pretrained.items():
                        if len(k.split('.')) >= 5:
                            new_pretrained['performer.net.blocks.{}.{}'.format(int(k.split('.')[3])*2, k.split('.', 4)[-1])] = pretrained[k]
                        else:
                            new_pretrained[k] = pretrained[k]
                else:
                    new_pretrained = pretrained
            else:
                for k, v in pretrained.items():
                    if len(k.split('.')) >= 5 and k.split('.')[4] == 'f':
                        new_pretrained['performer.net.layers.{}.0.{}'.format(k.split('.')[3], k.split('.', 6)[-1])] = pretrained[k]
                    elif len(k.split('.')) >= 5 and k.split('.')[4] == 'g':
                        new_pretrained['performer.net.layers.{}.{}.{}'.format(k.split('.')[3], 2 if lm_kwargs['cross_attend'] else 1, k.split('.', 6)[-1])] = pretrained[k]
                    else:
                        new_pretrained[k] = pretrained[k]
            lm.load_state_dict(new_pretrained, strict=False)
            print("Loaded pretrained language model: {}".format(pretrained_lm))
        self.lm = AutoregressiveWrapper(lm)
        self.dec = AutoregressiveWrapper(dec)

    @torch.no_grad()
    def generate(self, instrumental, lyrics_start, lyrics_len, vocals_start, vocals_len, **kwargs):
        enc_kwargs, lm_kwargs, dec_kwargs, kwargs = extract_and_set_enc_lm_dec_kwargs(kwargs)
        instrumental_encodings = self.enc(instrumental, return_encodings = True, **enc_kwargs)
        if self.lm_cross_attending:
            lm_kwargs.setdefault('context', instrumental_encodings)
            lm_kwargs.setdefault('context_mask', enc_kwargs['mask'])
        lyrics_encodings, lyrics = self.lm.generate(lyrics_start, lyrics_len, return_also_encodings = True, **{**lm_kwargs, **kwargs})
        vocals = self.dec.generate(vocals_start, vocals_len, context = instrumental_encodings, second_context = lyrics_encodings, **{**dec_kwargs, **kwargs})
        return lyrics, vocals

    def forward(self, instrumental, lyrics, vocals, **kwargs):
        enc_kwargs, lm_kwargs, dec_kwargs, kwargs = extract_and_set_enc_lm_dec_kwargs(kwargs)
        instrumental_encodings = self.enc(instrumental, return_encodings = True, **enc_kwargs)
        if self.lm_cross_attending:
            lm_kwargs.setdefault('context', instrumental_encodings)
            lm_kwargs.setdefault('context_mask', enc_kwargs['mask'])
        lyrics_encodings, lyrics_loss = self.lm(lyrics, return_also_encodings = True, **lm_kwargs)
        vocals_loss = self.dec(vocals, context = instrumental_encodings, second_context = lyrics_encodings, **dec_kwargs)
        return lyrics_loss + vocals_loss
        