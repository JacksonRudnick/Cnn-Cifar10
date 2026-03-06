from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as f
import pickle
import random
import numpy as np

# Hyper params
@dataclass
class cfg():
    width = 5
    depth = 3
    stride = 1
    padding = 2
    num_kernels = 2
    num_pools = 1
    batch_size = 10000
    num_batches = 5
    mini_batch_size = 10
    max_iters = 10
    learning_rate = 3e-4
    pool_size = 2
    device = ''
#############################

def unpickle(file) -> dict:
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_batch(split):
    if split == 'train':
        batch_num = random.randint(1,5)        
        data = unpickle(f'cifar-10-batches-py/data_batch_{batch_num}')
    else: 
        data = unpickle(f'cifar-10-batches-py/test_batch')

    indices = random.sample(range(cfg.batch_size), cfg.mini_batch_size)

    xs = torch.tensor(data[b'data'][indices], dtype=torch.float32)
    # Normalize the data to be between 0 and 1
    xs /= 255.0
    xs = xs.view(-1, 3, 32, 32)
    ys = torch.tensor([data[b'labels'][i] for i in indices], dtype=torch.long)

    xs = xs.to(cfg.device)
    ys = ys.to(cfg.device)

    return (xs, ys)
        
class Kernel(nn.Module):
    """Defines a Kernel/Filter to be applied on a volume"""
    def __init__(self, f: int, d: int, s: int, p: int):
        """
        Initialize a Kernel
        Creates a Linear Layer with (f x f x d) in and (1) out features

        Arguments:
            f --- Width or Height of Filter/Kernel
            d --- Depth of Filter/Kernel
            s --- Stride
            p --- Padding
        """
        super().__init__()
        
        self.width = f
        self.depth = d
        self.stride = s
        self.pad = p
        self.ln = nn.Linear(f*f*d, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 1000x3x32x32
        N, C, W, H = x.shape
        batch_results = []

        for n in range(N):
            #3x32x32
            inp = x[n]
            inp = f.pad(inp, [self.pad, self.pad, self.pad, self.pad], 'constant')

            out = torch.tensor((), device=cfg.device)
            # image p = 2 f = 5
            # r = 0 then r = s then r = 2s until r = 32 + (2*2) - 5 + 1 = 32
            for r in range(0, W + (2*self.pad) - self.width + 1, self.stride):
                # same as above but for c
                for c in range(0, W + (2*self.pad) - self.width + 1, self.stride):
                    # read into 32x32x3
                    # starting at [r][c] and going to [r+width][c+width]
                    # aka selecting the data the kernel should intake
                    # concatenate all data to create new activation map which should be size 28x28x1 without padding
                    out = torch.cat((out, self.ln(inp[:, r:r+self.width, c:c+self.width].flatten())))

            # With 2p 5f output is 32x32x1
            out = out.view(W, H)
            batch_results.append(out)

        return torch.stack(batch_results)

class Conv(nn.Module):
    """Defines a Convolutional layer containing a number of Kernels"""
    def __init__(self, n: int, s: int, f: int, d: int, p: int):
        """
        Create a Convolutional Layer
        Layer contains n Kernels with f width height, and d depth

        Arguments:
            n --- Number of Kernels
            s --- Stride
            f --- Width or Height of Kernel
            d --- Depth of Kernel
            p --- Padding of image
        """
        super().__init__()

        self.kernels = nn.ModuleList([Kernel(f, d, s, p) for _ in range(n)])

    def forward(self, x):
        #1000x3x32x32
        outs = [k(x) for k in self.kernels]

        out = torch.stack(outs, dim=1)

        out = f.relu(out)

        return out

class Net(nn.Module):
    """Defines the network that combines all the classes above"""
    def __init__(self, n: int, s: int, f: int, d: int, p: int):
        """
        Create the network
        """
        super().__init__()
        
        self.kernels = nn.ModuleList([
            Conv(cfg.num_kernels, cfg.stride, cfg.width, cfg.depth, cfg.padding),
            Conv(cfg.num_kernels, cfg.stride, cfg.width, cfg.num_kernels, cfg.padding)
            ])
        
        self.pools = nn.ModuleList([nn.MaxPool2d(cfg.pool_size) for _ in range(cfg.num_pools)])
        self.ln = nn.Linear(512, 10)

    def forward(self, x, targets=None):
        for i in range(len(self.kernels)):
            x = self.kernels[i](x)
            if int(i) != 0:
                x = self.pools[i-1](x)

        # 32x32x6 -> 28x28x6
        # 28x28x6 -> 24x24x6
        # 24 -> 20 -> 16

        x = x.view(x.size(0), -1)
        x = self.ln(x)

        if targets == None:
            return (x,)
        else:
            loss = f.cross_entropy(x, targets)

        return (x, loss)

def train():
    for iter in range(cfg.max_iters):
        # 1000x3x32x32
        xs, ys = get_batch('train')

        logits, loss = m(xs, ys)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

def test():
    xs, ys = get_batch('test')

    logits, loss = m(xs, ys)

    print(f"Test Loss: {loss.item()}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using CUDA")
        cfg.device = 'cuda'
    else:
        print("no cuda :(")
        exit()

    m = Net(cfg.num_kernels, cfg.stride, cfg.width, cfg.depth, cfg.padding)
    optimizer = torch.optim.Adam(m.parameters(), lr=cfg.learning_rate)
    m = m.to(cfg.device)

    train()
    test()