# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2022-05-21 19-38
@file: hybrid_inference.py
"""
import math

import yaml
import torch
import argparse
import numpy as np

# For reproducibility, comment these may speed up training
from src.asr import ASR
from src.audio import create_transform
from src.text import load_text_encoder
from src.util import feat_to_fig

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--config', type=str, help='Path to experiment config.')
parser.add_argument('--name', default=None, type=str, help='Name for logging.')
parser.add_argument('--logdir', default='log/', type=str,
                    help='Logging path.', required=False)
parser.add_argument('--ckpdir', default='ckpt/', type=str,
                    help='Checkpoint path.', required=False)
parser.add_argument('--outdir', default='result/', type=str,
                    help='Decode output path.', required=False)
parser.add_argument('--load', default=None, type=str,
                    help='Load pre-trained model (for training only)', required=False)
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed for reproducable results.', required=False)
parser.add_argument('--cudnn-ctc', action='store_true',
                    help='Switches CTC backend from torch to cudnn')
parser.add_argument('--njobs', default=6, type=int,
                    help='Number of threads for dataloader/decoding.', required=False)
parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
parser.add_argument('--no-pin', action='store_true',
                    help='Disable pin-memory for dataloader')
parser.add_argument('--test', action='store_true', help='Test the model.')
parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
parser.add_argument('--lm', action='store_true',
                    help='Option for training RNNLM.')
# Following features in development.
parser.add_argument('--amp', action='store_true', help='Option to enable AMP.')
parser.add_argument('--reserve-gpu', default=0, type=float,
                    help='Option to reserve GPU ram for training.')
parser.add_argument('--jit', action='store_true',
                    help='Option for enabling jit in pytorch. (feature in development)')
###
paras = parser.parse_args()
setattr(paras, 'gpu', not paras.cpu)
setattr(paras, 'pin_memory', not paras.no_pin)
setattr(paras, 'verbose', not paras.no_msg)
config = yaml.load(open(paras.config, 'r'), Loader=yaml.FullLoader)

np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(paras.seed)

device = "cpu"
ckpt = "ckpt/CTCAttChinaPopSong/latest.pth"
file = "/home/zliu-elliot/Desktop/testJiMei/f88_clip2.wav"

audio_transform, feat_dim = create_transform(config["data"]["audio"].copy())
# Text tokenizer
tokenizer = load_text_encoder(config["data"]["text"]["mode"], config["data"]["text"]["vocab_file"])
model = ASR(feat_dim, tokenizer.vocab_size, True, **config['model']).to(device)
ckpt = torch.load(ckpt, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()
audio = audio_transform(file)
audio_len = len(audio)
with torch.no_grad():
    ctc_output, encode_len, att_output, att_align, dec_state = model(audio.unsqueeze(0).to(device), torch.tensor([audio_len]).to(device), audio_len//4, emb_decoder=None)
att_output = att_output.argmax(dim=-1).squeeze(0).tolist()
att_output = tokenizer.decode(att_output, ignore_repeat=False)
ctc_output = ctc_output.argmax(dim=-1).squeeze(0).tolist()
ctc_output = tokenizer.decode(ctc_output, ignore_repeat=True)
attn_map = feat_to_fig(att_align[0, 0, :, :].cpu().detach())
pass




