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

MAX_LENGTH = 1024

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
save_path = "/root/2021080087/roformer_ann_2023/src/roformer.pt"

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

# data loading
test_path = "/root/2021080087/roformer_ann_2023/CAIL2019-SCM/test.json"
train_path = "/root/2021080087/roformer_ann_2023/CAIL2019-SCM/train.json"
valid_path = "/root/2021080087/roformer_ann_2023/CAIL2019-SCM/valid.json"
test_data = load_data(test_path)
train_data = load_data(train_path)
valid_data = load_data(valid_path)
best_valid_loss = float('inf')

def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def train(model, data, optimizer, loss_fn, epoch):
    model.train()
    train_losses = []
    total_loss = 0

    for batch_idx, batch in enumerate(data):
        optimizer.zero_grad()
        input_ids1 = []
        attention_mask1 = []
        input_ids2 = []
        attention_mask2 = []
        labels = []

        for sample in batch:
            input_ids1.append(sample['input_ids1'])
            attention_mask1.append(sample['attention_mask1'])
            input_ids2.append(sample['input_ids2'])
            attention_mask2.append(sample['attention_mask2'])
            labels.append(sample['label'])
        
        input_ids1 = torch.tensor(input_ids1).to(device)
        attention_mask1 = torch.tensor(attention_mask1).to(device)
        input_ids2 = torch.tensor(input_ids2).to(device)
        attention_mask2 = torch.tensor(attention_mask2).to(device)
        labels = torch.tensor(labels, dtype=torch.float).to(device)

        outputs1 = model(input_ids1, attention_mask=attention_mask1)
        outputs2 = model(input_ids2, attention_mask=attention_mask2)


         # bce loss
        probs1 = torch.softmax(outputs1.logits, dim=1).squeeze()
        probs2 = torch.softmax(outputs2.logits, dim=1).squeeze()
        # scores = torch.zeros(probs1.shape).to(device)

        # pdb.set_trace()

        # Calculate loss for each pair
        opposite = 1 - labels
        loss1 = loss_fn(probs1[:, 0], opposite)  # For A-B pairs
        loss2 = loss_fn(probs2[:, 0], labels)  # For A-C pairs

        # Total loss
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} Batch [{batch_idx+1}/{len(data)} ({int(100 * batch_idx / len(data))}%)]: Loss {loss.item()}")

    total_loss /= len(data)
    print(f"Epoch {epoch}: Average loss: {total_loss}")

def test(model, data, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data):
            input_ids1 = []
            attention_mask1 = []
            input_ids2 = []
            attention_mask2 = []
            labels = []

            for sample in batch:
                input_ids1.append(sample['input_ids1'])
                attention_mask1.append(sample['attention_mask1'])
                input_ids2.append(sample['input_ids2'])
                attention_mask2.append(sample['attention_mask2'])
                labels.append(sample['label'])
            
            input_ids1 = torch.tensor(input_ids1).to(device)
            attention_mask1 = torch.tensor(attention_mask1).to(device)
            input_ids2 = torch.tensor(input_ids2).to(device)
            attention_mask2 = torch.tensor(attention_mask2).to(device)
            labels = torch.tensor(labels, dtype=torch.float).to(device)

            outputs1 = model(input_ids1, attention_mask=attention_mask1)
            outputs2 = model(input_ids2, attention_mask=attention_mask2)

            # bce loss
            probs1 = torch.softmax(outputs1.logits, dim=1).squeeze()
            probs2 = torch.softmax(outputs2.logits, dim=1).squeeze()
            # scores = torch.zeros(probs1.shape).to(device)

            # Calculate loss for each pair
            opposite = 1 - labels
            loss1 = loss_fn(probs1[:, 0], opposite)  # For A-B pairs
            loss2 = loss_fn(probs2[:, 0], labels)  # For A-C pairs

            # Total loss
            both_loss = loss1 + loss2

            # Calculate BCE Loss
            # loss = loss_fn(scores, labels)
            test_loss += both_loss.item()
            total_samples += len(batch)

            # predict
            pred = probs1[:, 0] > probs2[:, 0]
            batch_correct = pred.eq(labels.view_as(pred)).sum().item()
            correct += batch_correct

            pdb.set_trace()


        test_loss /= total_samples
        accuracy = 100. * correct / total_samples

        if test_loss < best_valid_loss:
            best_valid_loss = test_loss
            torch.save(model.state_dict(), save_path)
        else:
            print("Validation loss did not decrease.")

        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)')
        pdb.set_trace()

def tokenize(dataset):
    tokenized_dataset = []
    for data in dataset:
        tokenized1 = tokenizer(
            text=data['A'],
            text_pair=data['B'],
            padding="max_length",
            truncation="only_first",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        tokenized2 = tokenizer(
            text=data['A'],
            text_pair=data['C'],
            padding="max_length",
            truncation="only_first",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
                
        # Assign binary label 0 or 1
        label = 0 if data['label'] == 'B' else 1

        combined_tokenized = {
            'input_ids1': tokenized1['input_ids'].squeeze().tolist(),
            # 'token_type_ids1': tokenized1['token_type_ids'],
            'attention_mask1': tokenized1['attention_mask'].squeeze().tolist(),
            'input_ids2': tokenized2['input_ids'].squeeze().tolist(),
            # 'token_type_ids2': tokenized2['token_type_ids'],
            'attention_mask2': tokenized2['attention_mask'].squeeze().tolist(),
            'label': label
        }
        tokenized_dataset.append(combined_tokenized)
    return tokenized_dataset

def tokenize_all():
    start_time = time.time()
    train_tokenized = tokenize(train_data)
    valid_tokenized = tokenize(valid_data)
    test_tokenized = tokenize(test_data)
    with open("train_tokenized.json", "w") as f:
        json.dump(train_tokenized, f)
    with open("valid_tokenized.json", "w") as f:
        json.dump(valid_tokenized, f)
    with open("test_tokenized.json", "w") as f:
        json.dump(test_tokenized, f)

    print(f"Tokenization Time: {int((time.time() - start_time) // 60)} minutes and {int((time.time() - start_time) % 60)} seconds")
    return train_tokenized, valid_tokenized, test_tokenized

def load_tokenized():
    start_time = time.time()
    with open("train_tokenized.json", "r") as f:
        train_tokenized = json.load(f)
    with open("valid_tokenized.json", "r") as f:
        valid_tokenized = json.load(f)
    with open("test_tokenized.json", "r") as f:
        test_tokenized = json.load(f)
    print(f"Loading Time: {int((time.time() - start_time) // 60)} minutes and {int((time.time() - start_time) % 60)} seconds")
    return train_tokenized, valid_tokenized, test_tokenized

# train
torch.cuda.empty_cache()
epochs = 5
lr = 2e-5
batch_size = 4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.BCELoss()

loading_tokenize = True
if loading_tokenize:
    print("Loading tokenized data...")
    train_tokenized, valid_tokenized, test_tokenized = load_tokenized()
else:   # if we don't have the tokenized data, tokenize it
    print("Tokenizing data...")
    train_tokenized, valid_tokenized, test_tokenized = tokenize_all()

start_time = time.time()

train_batches = list(batch_generator(train_tokenized, batch_size))
valid_batches = list(batch_generator(valid_tokenized, batch_size))
test_batches = list(batch_generator(test_tokenized, batch_size))

# train_small = train_batches[0:18]
valid_small = valid_batches[0:1]
# test_small = test_batches[0:4]

for epoch in range(1, epochs + 1):
    train(model, train_batches, optimizer, loss_fn, epoch)
    test(model, valid_small, loss_fn)

    print(f"Epoch {epoch}: {int((time.time() - start_time) // 60)} minutes and {int((time.time() - start_time) % 60)} seconds")
    if epoch % 5 == 0:
        test(model, test_batches, loss_fn)
    
torch.save(model.state_dict(), save_path)

