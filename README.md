TimeEncoding-pytorch
===
Pytorch implementation of Mercer's time encoding from paper "Self-attention with Functional Time Representation Learning" (https://arxiv.org/abs/1911.12864).

Usage
===
Time Encoding layer adds a new axis to the given tensor, representing every scalar in it as a high-dimensional vector.

```
from TimeEncoding import TimeEncoding as TE

TE_layer = TE(time_dim=4)
x = torch.rand((2,3))
x
>>> tensor([[0.1165, 0.9103, 0.6440],
        [0.7071, 0.6581, 0.4913]])

t = TE_layer(x)
t.shape
>>> torch.Size([2, 3, 4])
t
>>> tensor([[[ 0.0428, -0.9441,  0.3924, -0.8504],
             [ 0.3990, -0.9493,  0.3924, -0.8504],
             [ 0.8461, -0.9476,  0.3924, -0.8504]],
    
            [[ 0.7451, -0.9480,  0.3924, -0.8504],
             [ 0.8253, -0.9477,  0.3924, -0.8504],
             [ 0.9574, -0.9466,  0.3924, -0.8504]]])
```
