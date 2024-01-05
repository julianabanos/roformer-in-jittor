from transformers import RoFormerTokenizer
import jittor as jt
import json
from configuration_roformer import RoFormerConfig
from jt_model.jt_roformer import ModelSeqClassifier
import numpy as np
import pdb
import time
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
config.num_labels = 2

# set up tokenizer
tokenizer = RoFormerTokenizer.from_pretrained(vocab)

model = ModelSeqClassifier(config)

model.load(pretrained_model)

# data loading
test_path = "/root/2021080087/roformer_ann_2023/CAIL2019-SCM/test.json"
train_path = "/root/2021080087/roformer_ann_2023/CAIL2019-SCM/train.json"
valid_path = "/root/2021080087/roformer_ann_2023/CAIL2019-SCM/valid.json"
test_data = load_data(test_path)
train_data = load_data(train_path)
valid_data = load_data(valid_path)

# ideas
# a token between batchA and batchB

def train(model, data, optimizer, loss_fn, epoch):
    model.train()
    train_losses = []
    total_loss = 0

    for batch_idx, batch in enumerate(data):
        optimizer.zero_grad() # encode_plus?
        inputsAB = tokenizer(batch['A'] + tokenizer.sep_token + batch['B'], padding="max_length", return_tensors="np")
        inputsAC = tokenizer(batch['A'] + tokenizer.sep_token + batch['C'], padding="max_length", return_tensors="np")

        # process for jittor
        for key in inputsAB:
            inputsAB[key] = jt.Var(inputsAB[key])
        for key in inputsAC:
            inputsAC[key] = jt.Var(inputsAC[key])

        # label is B or C -> 0 or 1
        labels = jt.array([0 if label == 'B' else 1 for label in batch['label']])


        outputsAB = model(**inputsAB)
        outputsAC = model(**inputsAC)

        # pdb.set_trace()

        lossAB = loss_fn(outputsAB.logits, labels)
        lossAC = loss_fn(outputsAC.logits, labels)
        loss = (lossAB + lossAC) / 2

        # pdb.set_trace()

        optimizer.backward(loss)

        train_losses.append(loss.item())
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(batch), len(data),
                    100. * batch_idx / len(data), loss.item()))
    print("Average loss: ", total_loss / len(data))

def test(model, data, loss_fn, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with jt.no_grad():
        for batch_idx, batch in enumerate(data):
            inputsAB = tokenizer(batch['A'] + tokenizer.sep_token + batch['B'], padding="max_length", return_tensors="np")
            inputsAC = tokenizer(batch['A'] + tokenizer.sep_token + batch['C'], padding="max_length", return_tensors="np")

            # process for jittor
            for key in inputsAB:
                inputsAB[key] = jt.Var(inputsAB[key])
            for key in inputsAC:
                inputsAC[key] = jt.Var(inputsAC[key])

            # label is B or C -> 0 or 1
            labels = jt.array([0 if label == 'B' else 1 for label in batch['label']])

            outputsAB = model(**inputsAB)
            outputsAC = model(**inputsAC)

            lossAB = loss_fn(outputsAB.logits, labels)
            lossAC = loss_fn(outputsAC.logits, labels)
            loss = (lossAB + lossAC) / 2

            test_loss += loss.item()
            pred = outputsAB.logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(data)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(data),
            100. * correct / len(data)))
        

# train
epochs = 5
lr = 1e-5
batch_size = 8
optimizer = jt.optim.Adam(model.parameters(), lr=lr)
loss_fn = jt.nn.CrossEntropyLoss()
start_time = time.time()

for epoch in range(1, epochs + 1):
    train(model, train_data, optimizer, loss_fn, epoch)
    if epoch % 5 == 0:
        test(model, valid_data, loss_fn, epoch)
    print("Time elapsed: ", time.time() - start_time)

# save model
model.save("model.pt")
