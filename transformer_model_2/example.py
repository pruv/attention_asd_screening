import math
import time
import io

import torch
import torch.nn as nn

from torchtext.utils import download_from_url, extract_archive
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchsummary import summary
from torchviz import make_dot
import hiddenlayer as hl

from transformer_model_2.data_utils import batchify
from transformer_model_2.data_utils import data_process
from transformer_model_2.data_utils import data_process2
from transformer_model_2.data_utils import get_batch
from transformer_model_2.transformer2 import TransformerModel

url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
test_filepath, valid_filepath, train_filepath = extract_archive(download_from_url(url))
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer,
                                      iter(io.open(train_filepath,
                                                   encoding="utf8"))))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntokens = len(vocab.stoi)  # the size of vocabulary
emsize = 200  # embedding dimension
nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value
model = TransformerModel(ntoken=ntokens, ninp=emsize, nhead=nhead, nhid=nhid, nlayers=nlayers, device=device, dropout=dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

bptt = 35
train_data = data_process2(iter(io.open(train_filepath, encoding="utf8")), vocab, tokenizer)
val_data = data_process2(iter(io.open(valid_filepath, encoding="utf8")), vocab, tokenizer)
test_data = data_process2(iter(io.open(test_filepath, encoding="utf8")), vocab, tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size, device) # torch.Size([102499, 20]) 102499 senteces of each 20 words
val_data = batchify(val_data, eval_batch_size, device) # torch.Size([21441, 10])
test_data = batchify(test_data, eval_batch_size, device) # torch.Size([24185, 10])


def train():
    model.train()  # Turn on the train mode
    # print(model)
    # print(summary(model, input_size=[(20,), (2, 2)]))
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        if batch > 4:
            continue
        data, targets = get_batch(train_data, i, bptt)
        # data torch.Size([35, 20])
        # targets torch.Size([700])

        optimizer.zero_grad()
        if data.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)

        # print('data shape: ', data.shape) # torch.Size([35, 20])
        # print('src_mask shape: ', src_mask.shape) #  torch.Size([35, 35])

        output = model(data, src_mask)
        # output torch.Size([35, 20, 28783])
        # output = model(data)

        # print(summary(model, input_data=data, branching=True, verbose=2))
        # print(summary(model, [(20,)], dtypes=[torch.long]))
        # make_dot(output, params=dict(list(model.named_parameters()))).render("transformer_torchviz", format="png")

        output = output.view(-1, ntokens) # torch.Size([700, 28783])
        loss = criterion(output, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            if data.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, src_mask)
            # output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

best_val_loss = float("inf")
epochs = 3 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()