import torch


def data_process(raw_text_iter, vocab, tokenizer):
    data = [torch.tensor([vocab[token] for token in tokenizer(item)],
                         dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def data_process2(raw_text_iter, vocab, tokenizer):
    data = []
    for item in raw_text_iter:
        tokenized_lines = []
        for token in tokenizer(item):
            tokenized_lines.append(vocab[token])
        data.append(torch.tensor(tokenized_lines, dtype=torch.long))
    # data = [torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batchify(data, bsz, device):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, bptt): # source torch.Size([102499, 20])
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target
