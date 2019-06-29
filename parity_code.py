import torch

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



seq = generate_binary_seq()
parity_li = compute_parity_seq(seq)
print(seq)
print(parity_li)
