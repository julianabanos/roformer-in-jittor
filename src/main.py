from transformers import RoFormerTokenizer
tokenizer = RoFormerTokenizer('checkpoints/chinese_roformer_L-12_H-768_A-12/vocab.txt')
tokenizer.tokenize("今天。")
print(tokenizer.tokenize("天气。"))