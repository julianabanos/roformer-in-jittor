from transformers import RoFormerTokenizer
import jittor as jt
import json
from configuration_roformer import RoFormerConfig
from jt_model.jt_roformer import Model

# use cuda
jt.flags.use_cuda = 1

model_ckpt_path = "/root/2021080069/roformer_ann_2023/src/pytorch_dump/pytorch_model.bin"
config_path = "/root/2021080069/roformer_ann_2023/src/pytorch_dump/newconfig.json"

config = json.load(open(config_path))
# convert config into an object
config = type('', (), config)()

# set up tokenizer
tokenizer = RoFormerTokenizer('/root/2021080069/roformer_ann_2023/src/pytorch_dump/vocab.txt')


model = Model(config)
model.load(model_ckpt_path)
model.load_state_dict(jt.load(model_ckpt_path))

# test
input_ids = tokenizer.encode("今天天气不错")
input_ids = jt.array(input_ids).unsqueeze(0)
print(input_ids.shape)
output = model(input_ids)

print(output.shape)