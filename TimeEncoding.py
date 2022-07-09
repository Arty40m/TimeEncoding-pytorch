import torch
import numpy as np



class TimeEncoding(nn.Module):
    def __init__(self, time_dim, expand_dim=5):
        super().__init__()
        self.time_dim = time_dim
        self.expand_dim = expand_dim
        
        
        self.expand_coef = torch.nn.parameter.Parameter(
                        (torch.arange(expand_dim, dtype=torch.float32) + 1.).view(1, -1),  # 1, expand_dim
                         requires_grad=False)
            
        self.fr = torch.nn.parameter.Parameter(
                        torch.from_numpy(np.linspace(0, 8, time_dim).astype(np.float32)), 
                        requires_grad=True)
        
        self.V = torch.nn.parameter.Parameter(
                        torch.empty((time_dim, 2*expand_dim), dtype=torch.float32), 
                        requires_grad=True)
        
        self.B = torch.nn.parameter.Parameter(
                        torch.empty((time_dim,), dtype=torch.float32), 
                        requires_grad=True)
        
        
        torch.nn.init.xavier_uniform_(self.V, gain=1.0)
        torch.nn.init.zeros_(self.B)
        
        
    def forward(self, x):
        tile_shape = [1 for _ in range(len(x.shape))]
        x = torch.tile(x[..., None], (*tile_shape, self.time_dim)) # ... -> ..., time_dim
        x = x[..., None] # ..., time_dim -> ..., time_dim, 1
        
        period_var = 10.0 ** self.fr # time_dim
        period_var = period_var.view(self.time_dim, 1) # time_dim -> time_dim, 1
        period_var = torch.tile(period_var, (1, self.expand_dim)) # time_dim -> time_dim, expand_dim
        
        freq_var = 1. / period_var
        freq_var = freq_var * self.expand_coef
        freq_var = freq_var.view(*tile_shape, self.time_dim, self.expand_dim) # time_dim, expand_dim -> ..., time_dim, expand_dim
        
        x = x*freq_var # ..., time_dim, expand_dim
        sin = torch.sin(x)
        cos = torch.cos(x)
        
        time_enc = torch.cat([sin, cos], dim=-1) # ..., time_dim, 2 * expand_dim
        time_enc = time_enc*self.V
        time_enc = torch.sum(time_enc, dim=-1, keepdim=False) # ..., time_dim
        time_enc = time_enc + self.B
        
        return time_enc
        
        
        
        
