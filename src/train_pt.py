from transformers import RoFormerTokenizer, RoFormerForSequenceClassification
import jittor as jt
import json
from configuration_roformer import RoFormerConfig
from jt_model.jt_roformer import ModelSeqClassifier
import numpy as np
import pdb
import time
import jieba
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MAX_LENGTH = 512

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

# model = ModelSeqClassifier(config)
model = RoFormerForSequenceClassification.from_pretrained(pretrained_model, config=config_path)
model = model.to(device)

# model.load(pretrained_model)

# data loading
test_path = "/root/2021080087/roformer_ann_2023/CAIL2019-SCM/test.json"
train_path = "/root/2021080087/roformer_ann_2023/CAIL2019-SCM/train.json"
valid_path = "/root/2021080087/roformer_ann_2023/CAIL2019-SCM/valid.json"
test_data = load_data(test_path)
train_data = load_data(train_path)
valid_data = load_data(valid_path)

def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def train(model, data, optimizer, loss_fn, epoch):
    model.train()
    train_losses = []
    total_loss = 0

    for batch_idx, batch in enumerate(data):
        optimizer.zero_grad() # encode_plus?
        # Prepare batch data
        input_ids = []
        attention_mask = []
        labels = []

        for sample in batch:
            tokenized = tokenizer(
                sample['A'] + tokenizer.sep_token + sample['B'] + tokenizer.sep_token + sample['C'], 
                padding="max_length", 
                truncation=True,
                max_length=MAX_LENGTH,  # Define MAX_LENGTH according to your model's requirements
                return_tensors="pt"
            )
            input_ids.append(tokenized["input_ids"][0])
            attention_mask.append(tokenized["attention_mask"][0])
            labels.append(0 if sample['label'] == 'B' else 1)
        # inputsAC = tokenizer(batch['A'] + tokenizer.sep_token + batch['C'], padding="max_length", return_tensors="np")
            
        input_ids = torch.stack(input_ids).to(device)
        attention_mask = torch.stack(attention_mask).to(device)
        labels = torch.tensor(labels).to(device)

        outputs = model(input_ids, attention_mask=attention_mask)

        # pdb.set_trace()

        loss = loss_fn(outputs.logits, labels)

        # pdb.set_trace()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(batch), len(data)*batch_size,
                    100. * batch_idx / len(data), loss.item()))
    print("Average loss: ", total_loss / len(data))

def test(model, data, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data):
            # Prepare batch data
            input_ids = []
            attention_mask = []
            labels = []

            for sample in batch:
                tokenized = tokenizer(
                    sample['A'] + tokenizer.sep_token + sample['B'] + tokenizer.sep_token + sample['C'], 
                    padding="max_length", 
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                )
                input_ids.append(tokenized["input_ids"][0])
                attention_mask.append(tokenized["attention_mask"][0])
                labels.append(0 if sample['label'] == 'B' else 1)

            input_ids = torch.stack(input_ids).to(device)
            attention_mask = torch.stack(attention_mask).to(device)
            labels = torch.tensor(labels).to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            test_loss += loss.item()
            total_samples += len(batch)

            # Optionally calculate accuracy or other metrics here
            pred = outputs.logits.argmax(dim=1)
            correct += pred.eq(labels).sum().item()

        test_loss /= total_samples
        accuracy = 100. * correct / total_samples

        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.0f}%)')

# train
epochs = 5
lr = 2e-5
batch_size = 8
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()
start_time = time.time()

train_batches = list(batch_generator(train_data, batch_size))
valid_batches = list(batch_generator(valid_data, batch_size))
test_batches = list(batch_generator(test_data, batch_size))

# pdb.set_trace()
test(model, valid_batches, loss_fn)
for epoch in range(1, epochs + 1):
    train(model, train_batches, optimizer, loss_fn, epoch)
    # if epoch % 5 == 0:
    test(model, valid_batches, loss_fn)
    print("Minutes elapsed: ", (time.time() - start_time) / 60)

