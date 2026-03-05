from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as f
import pickle
import random

# Hyper params
@dataclass
class cfg():
    width = 5
    depth = 3
    stride = 1
    padding = 2
    num_kernels = 6
#############################

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_batch(split):
    if split == 'train':
        batch_num = random.randint(1,5)
        
        return unpickle(f'cifar-10-batches-py/data_batch_{batch_num}')
    return unpickle(f'cifar-10-batches-py/test_batch')
    
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

    def forward(self, x):
        # 32x32x3
        W, H, D = x.shape

        x = f.pad(x, [self.pad], 'constant')

        out = torch.tensor(())
        # image p = 2 f = 5
        # r = 0 then r = s then r = 2s until r = 32 + (2*2) - 5 + 1 = 32
        for r in range(0, W + (2*self.pad) - self.width + 1, self.stride):
            # same as above but for c
            for c in range(0, W + (2*self.pad) - self.width + 1, self.stride):
                # read into 32x32x3
                # starting at [r][c] and going to [r+width][c+width]
                # aka selecting the data the kernel should intake
                # concatenate all data to create new activation map which should be size 28x28x1 without padding
                out = torch.cat((out, self.ln(x[r:r+self.width, c:c+self.width].flatten())))
        
        return out
    
class Conv(nn.Module):
    """Defines a Convolutional layer containing a number of Kernels"""
    def __init__(self, n: int, s: int, f: int, d: int, p: int, do_pooling=True):
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
        
        if do_pooling:
            self.do_pooling = True
            self.pools = nn.MaxPool3d(kernel_size=f)
        else:
            self.do_pooling = False

        self.kernels = nn.ModuleList([Kernel(f, d, s, p) for _ in range(n)])

    def forward(self, x):
        out = torch.tensor(())

        for k in self.kernels:
            out = torch.cat((out, k(x)))

        out = f.relu(out)

        if self.do_pooling:
            out = self.pools(out)

        return out

