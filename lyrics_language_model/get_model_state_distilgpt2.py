# porting code from https://github.com/tenexcoder/huggingface-tutorials/blob/main/performer/hf_port.py
import collections
import torch
from transformers import GPT2LMHeadModel


model = GPT2LMHeadModel.from_pretrained('distilgpt2')
model_state = model.state_dict()

num_tokens = model.transformer.wte.num_embeddings
dim = model.transformer.wte.embedding_dim
max_seq_len = model.transformer.wpe.num_embeddings
depth = len(list(model.transformer.h.children()))

performer_state = collections.OrderedDict()

# emb
performer_state['token_emb.weight'] = model_state['transformer.wte.weight']
performer_state['pos_emb.emb.weight'] = model_state['transformer.wpe.weight']

# layers norm
performer_state['norm.weight'] = model_state['transformer.ln_f.weight']
performer_state['norm.bias'] = model_state['transformer.ln_f.bias']

attn_id = ['to_q', 'to_k', 'to_v']
for layer_idx in range(depth):
    from_layer = 'transformer.h.' + str(layer_idx)
    to_layer = 'performer.net.blocks.' + str(layer_idx)

    # attn norm
    performer_state[to_layer + '.f.net.norm.weight'] = \
        model_state[from_layer + '.ln_1.weight']
    performer_state[to_layer + '.f.net.norm.bias'] = \
        model_state[from_layer + '.ln_1.bias']

    # load 'from' qkv weight and bias - qkv_w and qkv_b
    # then split into lists [q_w, k_w, v_w] and [q_b, k_b, v_b]
    qkv_w = model_state[from_layer + '.attn.c_attn.weight'].split(dim, dim=1)
    qkv_b = model_state[from_layer + '.attn.c_attn.bias'].split(dim, dim=0)

    # set 'to' qkv weight and bias - q_w, q_b, k_w, k_b. v_w, v_b
    for idx, (w, b) in enumerate(zip(qkv_w, qkv_b)):
        performer_state[to_layer + '.f.net.fn.' + attn_id[idx] + '.weight'] = w
        performer_state[to_layer + '.f.net.fn.' + attn_id[idx] + '.bias'] = b

    # attn projection
    performer_state[to_layer + '.f.net.fn.to_out.weight'] = \
        model_state[from_layer + '.attn.c_proj.weight']
    performer_state[to_layer + '.f.net.fn.to_out.bias'] = \
        model_state[from_layer + '.attn.c_proj.bias']

    # mlp norm
    performer_state[to_layer + '.g.net.norm.weight'] = \
        model_state[from_layer + '.ln_2.weight']
    performer_state[to_layer + '.g.net.norm.bias'] = \
        model_state[from_layer + '.ln_2.bias']

    # mlp - transpose 'from' to match 'to' shape
    # TODO: check why 'weight' shape isn't 1 to 1
    performer_state[to_layer + '.g.net.fn.fn.w1.weight'] = \
        torch.einsum('ij->ji', model_state[from_layer + '.mlp.c_fc.weight'])
    performer_state[to_layer + '.g.net.fn.fn.w1.bias'] = \
        model_state[from_layer + '.mlp.c_fc.bias']
    performer_state[to_layer + '.g.net.fn.fn.w2.weight'] = \
        torch.einsum('ij->ji', model_state[from_layer + '.mlp.c_proj.weight'])
    performer_state[to_layer + '.g.net.fn.fn.w2.bias'] = \
        model_state[from_layer + '.mlp.c_proj.bias']

# decoder head
performer_state['to_out.weight'] = \
    model_state['lm_head.weight']

torch.save(performer_state, './distilgpt2.pt')
print('done porting')