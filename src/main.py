from transformers import RoFormerTokenizer
import jittor as jt
import json
from configuration_roformer import RoFormerConfig
from jt_model.jt_roformer import ModelCausalLM
import numpy as np
import pdb
# use cuda
jt.flags.use_cuda = 1

pretrained_model = "/root/2021080087/roformer_ann_2023/src/roformer_chinese_sim_char_base/pytorch_model.bin"
config_path = "/root/2021080087/roformer_ann_2023/src/roformer_chinese_sim_char_base/config.json"
vocab = "/root/2021080087/roformer_ann_2023/src/roformer_chinese_sim_char_base/vocab.txt"

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


# set up tokenizer
# tokenizer = RoFormerTokenizer('/root/2021080069/roformer_ann_2023/src/pytorch_dump/vocab.txt')
tokenizer = RoFormerTokenizer.from_pretrained(vocab)

model = ModelCausalLM(config)


print(config.is_decoder)
# model = Model(config)
model.load(pretrained_model)
# model.load_state_dict(jt.load(model_ckpt_path))


model.eval()


def gen_synonyms(text, n=100, k=20):
    ''''含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    '''
    # 寻找所有相似的句子
    r = []
    inputs1 = tokenizer(text, return_tensors="pt")
    # inputs1 to numpy
    
    for key in inputs1:
        inputs1[key] = inputs1[key].numpy()
        inputs1[key] = jt.Var(inputs1[key])

    for _ in range(n):
        print("Generating... ", _ , "/", n)
        generated = model.generate(**inputs1, top_p=0.95, do_sample=True, max_length=128)
        # print("generated", generated)
        output = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        output = output.replace(" ","").replace(text, "") # 去除空格，去除原始text文本。
        # print(output)
        r.append(output)
    
    
    # 对相似的句子进行排序
    r = [i for i in set(r) if i != text and len(i) > 0]
    r = [text] + r
    inputs2 = tokenizer(r, padding=True, return_tensors="pt")
    # pdb.set_trace()
    for key in inputs2:
        inputs2[key] = inputs2[key].numpy()
        inputs2[key] = jt.Var(inputs2[key])
    with jt.no_grad():
        outputs = model(**inputs2)            
        Z = outputs.pooler_output.cpu().numpy()
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    
    return [r[i + 1] for i in argsort[:k]]


out = gen_synonyms("广州和深圳哪个好？")
print(out)
# test
# input_ids = tokenizer.encode("今天天气不错")
# input_ids = jt.array(input_ids).unsqueeze(0)
# print(input_ids.shape)
# print(input_ids)
# output = model(input_ids=input_ids)

# print(output.shape)