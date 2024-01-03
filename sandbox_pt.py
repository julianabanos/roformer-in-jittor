# pytorch sandbox for loading checkpoints
import torch
import json
from transformers import BertModel, BertConfig

config = "/root/2021080087/roformer_ann_2023/checkpoints/chinese_roformer_L-12_H-768_A-12/bert_config.json"

# read config
with open(config, "r", encoding="utf-8") as f:
    config = json.load(f)

print(f"Building PyTorch model from configuration: {config}")

ckpt = "/root/2021080087/roformer_ann_2023/checkpoints/chinese_roformer_L-12_H-768_A-12/pytorch_dump/pytorch_model.bin"

# load checkpoint
state_dict = torch.load(ckpt, map_location="cpu")

bert_config = BertConfig.from_dict(config)

model = BertModel(bert_config)

# load state_dict
model.load_state_dict(state_dict, strict=False)

# print model
print(model)
