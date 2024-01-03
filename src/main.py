from transformers import RoFormerTokenizer
from modeling_jt_roformer import RoFormerModel, RoFormerConfig
import jittor as jt
tokenizer = RoFormerTokenizer('src/pytorch_dump/vocab.txt')
# tokenizer.tokenize("今天。")
# print(tokenizer.tokenize("天气。"))

# use cuda
jt.flags.use_cuda = 1

model_ckpt_path = "src/pytorch_dump/pytorch_model.bin"
config_path = "src/pytorch_dump/config.json"
config = RoFormerConfig.from_json_file(config_path)
model = RoFormerModel(config)
model.load_state_dict(jt.load(model_ckpt_path), strict=False)

# test
input_ids = tokenizer.encode("今天天气不错")
input_ids = jt.array(input_ids).unsqueeze(0)
print(input_ids.shape)
output = model(input_ids)

print(output.shape)