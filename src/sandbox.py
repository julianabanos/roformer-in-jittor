from transformers import RoFormerTokenizer
import jittor as jt
import json
from configuration_roformer import RoFormerConfig
from jt_model.jt_roformer import ModelSeqClassifier
import numpy as np
import pdb
import time
import jieba

MAX_LENGTH = 512

# use cuda
jt.flags.use_cuda = 1

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data

pretrained_model = "/root/2021080087/roformer_ann_2023/src/roformer_chinese_base/pytorch_model.bin"
config_path = "/root/2021080087/roformer_ann_2023/src/roformer_chinese_base/config.json"
vocab = "/root/2021080087/roformer_ann_2023/src/roformer_chinese_base/vocab.txt"

config = json.load(open(config_path))
# convert config into an object
config = type('', (), config)()
config.is_decoder = True
config.add_cross_attention = True
config.chunk_size_feed_forward = 0
config.add_pooling_layer = True
config.norm_type = "layer_norm"
config.use_bias = True
config.rotary_value = False
config.use_cache = True
config.num_labels = 2

# set up tokenizer
tokenizer = RoFormerTokenizer.from_pretrained(vocab)

model = ModelSeqClassifier(config)

model.load(pretrained_model)
model.eval()