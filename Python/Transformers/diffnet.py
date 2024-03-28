import torch
from torch import nn
import copy
import numpy as np
import math
import torch.nn.functional as F

def clone_layers(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

class PositionalEncoder(nn.Module):
  '''
    Generate positional encodings used in the relative multi-head attention module.
    These encodings are the same as the original transformer model: https://arxiv.org/abs/1706.03762

    Parameters:
      max_len (int): Maximum sequence length (time dimension)

    Inputs:
      len (int): Length of encodings to retrieve
    
    Outputs
      Tensor (len, d_model): Positional encodings
  '''
  def __init__(self, d_model, max_len=10000):
    super(PositionalEncoder, self).__init__()
    self.d_model = d_model
    encodings = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float)
    inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
    encodings[:, 0::2] = torch.sin(pos[:, None] * inv_freq)
    encodings[:, 1::2] = torch.cos(pos[:, None] * inv_freq)
    self.register_buffer('encodings', encodings)
    
  def forward(self, len):
      return self.encodings[:len, :]

class RelativeMultiHeadAttention(nn.Module):
  '''
    Relative Multi-Head Self-Attention Module. 
    Method proposed in Transformer-XL paper: https://arxiv.org/abs/1901.02860

    Parameters:
      d_model (int): Dimension of the model
      num_heads (int): Number of heads to split inputs into
      dropout (float): Dropout probability
      positional_encoder (nn.Module): PositionalEncoder module
    
    Inputs:
      x (Tensor): (batch_size, time, d_model)
      mask (Tensor): (batch_size, time, time) Optional mask to zero out attention score at certain indices
    
    Outputs:
      Tensor (batch_size, time, d_model): Output tensor from the attention module.
  
  '''
  def __init__(self, d_model=144, num_heads=4, dropout=0.1, positional_encoder=PositionalEncoder(144)):
    super(RelativeMultiHeadAttention, self).__init__()

    #dimensions
    assert d_model % num_heads == 0
    self.d_model = d_model
    self.d_head = d_model // num_heads
    self.num_heads = num_heads

    # Linear projection weights
    self.W_q = nn.Linear(d_model, d_model)
    self.W_k = nn.Linear(d_model, d_model)
    self.W_v = nn.Linear(d_model, d_model)
    self.W_pos = nn.Linear(d_model, d_model, bias=False)
    self.W_out = nn.Linear(d_model, d_model)

    # Trainable bias parameters
    self.u = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
    self.v = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
    torch.nn.init.xavier_uniform_(self.u)
    torch.nn.init.xavier_uniform_(self.v)

    # etc
    self.layer_norm = nn.LayerNorm(d_model, eps=6.1e-5)
    self.positional_encoder = positional_encoder
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask=None):
    batch_size, seq_length, _ = x.size()

    #layer norm and pos embeddings
    x = self.layer_norm(x)
    pos_emb = self.positional_encoder(seq_length)
    pos_emb = pos_emb.repeat(batch_size, 1, 1)

    #Linear projections, split into heads
    q = self.W_q(x).view(batch_size, seq_length, self.num_heads, self.d_head)
    k = self.W_k(x).view(batch_size, seq_length, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, num_heads, d_head, time)
    v = self.W_v(x).view(batch_size, seq_length, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, num_heads, d_head, time)
    pos_emb = self.W_pos(pos_emb).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, num_heads, d_head, time)

    #Compute attention scores with relative position embeddings
    AC = torch.matmul((q + self.u).transpose(1, 2), k)
    BD = torch.matmul((q + self.v).transpose(1, 2), pos_emb)
    BD = self.rel_shift(BD)
    attn = (AC + BD) / math.sqrt(self.d_model)

    #Mask before softmax with large negative number
    if mask is not None:
      mask = mask.unsqueeze(1)
      mask_value = -1e+30 if attn.dtype == torch.float32 else -1e+4
      attn.masked_fill_(mask, mask_value)

    #Softmax
    attn = F.softmax(attn, -1)

    #Construct outputs from values
    output = torch.matmul(attn, v.transpose(2, 3)).transpose(1, 2) # (batch_size, time, num_heads, d_head)
    output = output.contiguous().view(batch_size, -1, self.d_model) # (batch_size, time, d_model)

    #Output projections and dropout
    output = self.W_out(output)
    return self.dropout(output)


  def rel_shift(self, emb):
    '''
      Pad and shift form relative positional encodings. 
      Taken from Transformer-XL implementation: https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py 
    '''
    batch_size, num_heads, seq_length1, seq_length2 = emb.size()
    zeros = emb.new_zeros(batch_size, num_heads, seq_length1, 1)
    padded_emb = torch.cat([zeros, emb], dim=-1)
    padded_emb = padded_emb.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
    shifted_emb = padded_emb[:, :, 1:].view_as(emb)
    return shifted_emb
  
class ConvBlock(nn.Module):

    def __init__(self,inp=64,dropout=0.3,kernel_size=31):

        super().__init__()

        self.conv =  nn.Sequential(
            nn.Conv1d(inp,inp*2,kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(inp,inp,kernel_size=kernel_size,padding="same",groups=inp),
            nn.BatchNorm1d(inp),
            nn.SiLU(),
            nn.Conv1d(inp,inp,kernel_size=1),
            nn.Dropout(dropout)
        )
        self.ln = nn.LayerNorm(inp)


    def forward(self,x):
        x = self.ln(x)
        x = x.transpose(1,2)
        x = self.conv(x)
        x = x.transpose(2,1)
        return x


class FFN(nn.Module):

    def __init__(self,inp=64,dropout=0.3):

        super().__init__()

        self.ffn = nn.Sequential(
            nn.LayerNorm(inp),
            nn.Linear(inp,inp*4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inp*4,inp),
            nn.Dropout(dropout)
        )

    
    def forward(self,x):

        x = self.ffn(x)
        return x
    


class Conformer_block(nn.Module):

    def __init__(self,inp=64,dropout=0.3,kernel_size=31,nheads=2):
        super().__init__()

        self.ffn1 = FFN(inp=inp,dropout=dropout)
        self.conv = ConvBlock(kernel_size=kernel_size,dropout=dropout)
        self.ln = nn.LayerNorm(inp)
        self.mha = RelativeMultiHeadAttention(inp, nheads, dropout, PositionalEncoder(64))
        self.ffn2 = FFN(inp=inp,dropout=dropout)
        self.ln2 = nn.LayerNorm(inp)
    
    def forward(self,x):
        
        out = x + 0.5*self.ffn1(x)
        out = out + self.mha(out,mask=None)
        out = out + self.conv(out)
        out = out + 0.5*self.ffn2(out)
        out = self.ln(out)

        return out



class EEG_Conformer(nn.Module):

    def __init__(self,nblock=5,inp=64,kernel_size=31,nheads=4,dropout=0.4):
        super().__init__()

        self.confblock = nn.ModuleList([Conformer_block(inp=inp,kernel_size=kernel_size,nheads=nheads,dropout=dropout) for _ in range(nblock)])
        self.lin = nn.Linear(inp,1)
    
    def forward(self,x):
        x = x.transpose(1,2)
        for conf in self.confblock:
            x = conf(x)
        x = self.lin(x)
        return x.squeeze(-1)


    
