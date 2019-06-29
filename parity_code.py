import torch
from torch import nn
from torch import optim

import random

random.seed(0)


def generate_binary_seq(seq_len=20):
    alphabet = list(range(10))
    return [random.choice(alphabet) for i in range(seq_len)]

def compute_parity_seq(seq):
    parity_li = []
    num_even = 0
    for _, e in enumerate(seq):
        if e % 2 == 0:
            num_even += 1
    
        if num_even % 2 == 1:
            parity_li.append(1)
        else:
            parity_li.append(0)
    
    return parity_li

seq_li = [torch.LongTensor(generate_binary_seq()) for _ in range(100)]
parity_li = [torch.LongTensor(compute_parity_seq(e)) for e in seq_li]

# seq = generate_binary_seq()
# parity_li = compute_parity_seq(seq)
# print(seq)
# print(parity_li)

class ParityCheck(nn.Module):

    def __init__(self):
        super(ParityCheck, self).__init__()
        self.encoding = nn.Embedding(10, 8)
        self.lstm = nn.LSTM(8, 4)
        self.linear = nn.Linear(4, 2)
    
    def forward(self, x):
        encoding = self.encoding(x)
        encoding = encoding.view(len(x), 1, -1)
        out, (h_n, c_n) = self.lstm(encoding)
        output = self.linear(out)
        
        return output.squeeze()


parity_check = ParityCheck()
sgd = optim.Adam(parity_check.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(500):

    for i, seq in enumerate(seq_li[:1]):
        sgd.zero_grad()
        output = parity_check(seq)
        loss = criterion(output, parity_li[i])
        loss.backward()
        print(f'{epoch} {i}', loss.item())
        sgd.step()