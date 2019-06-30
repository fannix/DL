import torch
from torch import nn
from torch import optim
import numpy as np
import torchtext

from collections import Counter
import random
import string
random.seed(0)

def generate_seq(len=10):
    return [random.choice(string.ascii_uppercase) for _ in range(len)]


def get_reverse(seq):
    return list(reversed(seq))

def to_tensor(seq_li):
    return [torch.LongTensor([vocab.stoi[e] for e in seq]) for seq in seq_li]

vocab = torchtext.vocab.Vocab(
    Counter(string.ascii_uppercase + string.digits), specials=['{'])

train_seq_li = [generate_seq() for _ in range(1000)]
train_rev_li = [get_reverse(e) for e in train_seq_li]

train_seq_tensor_li = to_tensor(train_seq_li)
train_rev_tensor_li = to_tensor(train_rev_li)

val_seq_li = [generate_seq() for _ in range(1000)]
val_rev_li = [get_reverse(e) for e in val_seq_li]

val_seq_tensor_li = to_tensor(val_seq_li)
val_rev_tensor_li = to_tensor(val_rev_li)

class Encoder(nn.Module):

    def __init__(self, embed_size, hidden_size):

        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(len(vocab.stoi), embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        encoding = self.embedding(x).view(len(x), 1, -1)
        lstm, context = self.lstm(encoding)
        output = self.linear(lstm)

        return output


class Decoder(nn.Module):

    def __init__(self, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(len(vocab.stoi), embed_size)
        self.lstm=nn.GRU(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, len(vocab.stoi))
        self.init_context()
    
    def forward(self, x):
        encoding = self.embedding(x).view(len(x), 1, -1)
        lstm, self.context = self.lstm(encoding, self.context)
        output = self.linear(lstm)

        return output

    def init_context(self, context = None):
        self.context = context


class EncodeDecoder(nn.Module):

    def __init__(self, encoder, decoder):
        super(EncodeDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        encoded = self.encoder(x)[-1]
        self.decoder.init_context(encoded.view(1, 1, -1))

        prev_token = torch.LongTensor([0])

        output = torch.zeros(len(x), len(vocab.stoi))
        for i in range(len(x)):
            decoded = self.decoder(prev_token)
            output[i] = decoded
            prev_token = decoded.topk(1)[1]
        
        return output


encoder = Encoder(8, 4)
decoder = Decoder(8, 4)
endecoder = EncodeDecoder(encoder, decoder)

criterion = nn.CrossEntropyLoss()
encode_optimizer = optim.Adam(encoder.parameters())
decode_optimizer = optim.Adam(decoder.parameters())
for epoch in range(5001):
    encode_optimizer.zero_grad()
    decode_optimizer.zero_grad()
    output = endecoder(train_seq_tensor_li[0])
    loss = criterion(output, train_rev_tensor_li[0])
    if epoch % 100 == 0:
        print(f"{epoch}", loss.item())
    loss.backward()
    encode_optimizer.step()
    decode_optimizer.step()
