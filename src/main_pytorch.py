import torch
import numpy as np
from configuration_roformer import RoFormerConfig
from modeling_roformer import RoFormerForCausalLM
from transformers import RoFormerTokenizer#, RoFormerForCausalLM, RoFormerConfig
import pdb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pretrained_model = "/root/2021080087/roformer_ann_2023/src/roformer_chinese_sim_char_base/pytorch_model.bin"
config_path = "/root/2021080087/roformer_ann_2023/src/roformer_chinese_sim_char_base/config.json"
vocab = "/root/2021080087/roformer_ann_2023/src/roformer_chinese_sim_char_base/vocab.txt"
tokenizer = RoFormerTokenizer.from_pretrained(vocab)

config = RoFormerConfig.from_pretrained(config_path)
config.is_decoder = True
config.eos_token_id = tokenizer.sep_token_id
config.pooler_activation = "linear"
# pdb.set_trace()
model = RoFormerForCausalLM.from_pretrained(pretrained_model, config=config)
model.to(device)
model.eval()

def gen_synonyms(text, n=1
, k=20):
    ''''含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    '''
    # 寻找所有相似的句子
    r = []
    inputs1 = tokenizer(text, return_tensors="pt")

    for _ in range(n):
        print("Generating... ", _ , "/", n)
        inputs1.to(device)
        # generation = model.generate(**inputs1, top_p=0.95, do_sample=True, max_length=128)
        
        generation2 = model.generate2(**inputs1, top_p=0.95, do_sample=True, max_length=128)
        
        # output = tokenizer.batch_decode(generation, skip_special_tokens=True)[0].replace(" ","").replace(text, "") # 去除空格，去除原始text文本。
        output2 = tokenizer.batch_decode(generation2, skip_special_tokens=True)[0].replace(" ","").replace(text, "") # 去除空格，去除原始text文本。
        # print("generation: ", output)
        # print("generation2: ", output2)
        # pdb.set_trace()
        r.append(output2)

    # pdb.set_trace()     

    # 对相似的句子进行排序
    r = [i for i in set(r) if i != text and len(i) > 0]
    r = [text] + r
    inputs2 = tokenizer(r, padding=True, return_tensors="pt")
    # pdb.set_trace()
    with torch.no_grad():
        inputs2.to(device)
        outputs = model(**inputs2)
        Z = outputs.pooler_output.cpu().numpy()
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()    
    return [r[i + 1] for i in argsort[:k]]


out = gen_synonyms("广州和深圳哪个好？")
print(out)