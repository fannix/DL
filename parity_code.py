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

seq_li = [torch.LongTensor(generate_binary_seq()) for _ in range(1000)]
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

for epoch in range(20):

    running_loss = 0
    n_info = 100
    for i, seq in enumerate(seq_li):
        sgd.zero_grad()
        output = parity_check(seq)
        loss = criterion(output, parity_li[i])
        running_loss += loss.item()
        if i % n_info == n_info - 1 and (epoch == 0 or epoch % 10 == 9):
            print(f'{epoch} {i}', running_loss / n_info)
            running_loss = 0

        loss.backward()

        sgd.step()

random.seed()

val_seq_li = [torch.LongTensor(generate_binary_seq()) for _ in range(100)]
val_parity_li = [torch.LongTensor(compute_parity_seq(e)) for e in val_seq_li]


def validate():
    running_loss = 0
    n_info = 10
    n_correct = 0
    for i, seq in enumerate(val_seq_li):
        output = parity_check(seq)
        loss = criterion(output, val_parity_li[i])
        running_loss += loss.item()
        if i % n_info == n_info - 1:
            print(f'{i}', running_loss / n_info)
            running_loss = 0
        
        n_correct += torch.sum(torch.argmax(output, 1) == val_parity_li[i]).item()
    
    print(n_correct / (len(val_parity_li) * len(val_parity_li[0])))


validate()

all_even = torch.LongTensor([2] * 40)
output = parity_check(all_even)
torch.argmax(output, 1)

all_odd = torch.LongTensor([1] * 20)
output = parity_check(all_odd)
torch.argmax(output, 1)