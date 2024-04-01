import torch
from torch import nn
import copy
import numpy as np
from positional_encodings.torch_encodings import PositionalEncoding1D

def clone_layers(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

class CNN(torch.nn.Module):

    def __init__(self,num_eeg_channels=64,num_convs=3,filter_size=32,kernel_size=8,dropout=0.5,times=256):

        super().__init__()

        self.n_convs = num_convs
        self.dropout = dropout
        self.filt_size = filter_size
        self.kernel_size = kernel_size
        self.num_eeg_channels = num_eeg_channels
        self.conv1 = nn.Conv1d(self.num_eeg_channels,self.filt_size,self.kernel_size,padding="same")
        self.conv_layers = clone_layers(nn.Conv1d(self.filt_size,self.filt_size,self.kernel_size,padding="same"),self.n_convs-1)
        self.times = times

        self.norm = clone_layers(nn.LayerNorm([self.filt_size,self.times]),self.n_convs-1)
        self.drop = clone_layers(nn.Dropout(self.dropout),self.n_convs-1)
        self.act = clone_layers(nn.GELU(),self.n_convs-1)

    def forward(self,eeg):
        
        out = self.conv1(eeg)

        for conv,norm,drop,act in zip(self.conv_layers,self.norm,self.drop,self.act):
             out = drop(norm(act(conv(out))))

        return out
    

        
class TransEncoder(torch.nn.Module):

    def __init__(self,hidden=32,num_layers=2,dropout=0.5,nhead=4,dff = 32,times=192):

        super().__init__()

        self.hidden = hidden
        self.n_layers = num_layers
        self.dropout = dropout
        self.nhead = nhead
        self.dff = dff
        self.pos = PositionalEncoding1D(self.hidden)
        self.alpha = nn.Parameter(torch.ones(1))


        self.encoder_layers = nn.TransformerEncoderLayer(self.hidden,nhead=self.nhead,dim_feedforward=self.dff,batch_first=True,activation="gelu",dropout=self.dropout)
        self.Encoder = nn.TransformerEncoder(self.encoder_layers,self.n_layers,norm=nn.LayerNorm(self.hidden,times))

    
    def forward(self,eeg):

        pos = self.pos(eeg)
        eeg = eeg + self.alpha*pos
        out = self.Encoder(eeg)

        return out


class Encoder_Prenet(nn.Module):

    def __init__(self,nblock,lin_out,preksize=8,nconvs=5,filter_size=32,kernel_size=32,times=192):

        super().__init__()

        self.nblock = nblock
        self.lin_out = lin_out
        self.ksize = preksize

        self.Convolutions = clone_layers(CNN(num_convs=nconvs,filter_size=filter_size,kernel_size=preksize,dropout=0.5,times=times),self.nblock)
        self.linear = clone_layers(nn.Linear(filter_size,self.lin_out),self.nblock)
        self.convout = clone_layers(nn.Conv1d(self.lin_out,64,self.ksize,padding="same"),self.nblock)
        self.act = clone_layers(nn.GELU(),self.nblock)


    def forward(self,eeg):

        for conv_in,lin,conv_out,act in zip(self.Convolutions,self.linear,self.convout,self.act):

            out = conv_in(eeg)
            out = out.permute((0,2,1))
            out = lin(out)
            out = out.permute((0,2,1))
            out = conv_out(out)
            out = act(out)
            nshape = out.shape[1:]
            out = nn.LayerNorm(nshape).to(out.device)(out)
            eeg = eeg+out
        
        return eeg
    
class model(nn.Module):

    def __init__(self,nblock=3,lin_out=16,kernel_size=15,transformer_hidden=128,n_layers=4,n_heads=4,dff=64,filter_size=32,nconvs=4,encpre = Encoder_Prenet,enc = TransEncoder,times=192):

        super().__init__()

        self.Enc_Pre = encpre(nblock=nblock,lin_out=lin_out,kernel_size=kernel_size,nconvs=nconvs,filter_size=filter_size,times=times)
        self.Encoder = enc(hidden=transformer_hidden,num_layers=n_layers,dropout=0.5,nhead=n_heads,dff = dff,times=times)
        self.int_conv = nn.Conv1d(64,self.Encoder.hidden,1,padding="same")
        self.lin = nn.LazyLinear(times)
        self.lin2 = nn.LazyLinear(1024)

        
    def forward(self,eeg):

        out = self.Enc_Pre(eeg)
        out = self.int_conv(out)
        out = out.permute((0,2,1))
        out = self.Encoder(out)
        out = nn.Flatten()(out)
        out = self.lin2(out)
        out = self.lin(out)

        return out.squeeze(-1)


class CNN2(torch.nn.Module):

    def __init__(self,k=32,kernel=(1,25),stride=(1,1)):
        super().__init__()

        self.k = k
        self.kernel = kernel
        self.stride = stride
        self.conv1 = torch.nn.Conv2d(1,self.k,self.kernel,stride=self.stride)
        self.conv2 = torch.nn.Conv2d(self.k,self.k,(64,1),stride=self.stride)
        self.avg = torch.nn.AvgPool2d(kernel_size=(1,15),stride=(1,5))
    
    def forward(self,x):
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.avg(out)
        out = torch.reshape(out,[out.shape[0],out.shape[-1],out.shape[1]])
        return out
    

    
class out_model(torch.nn.Module):

    def __init__(self,hid=32):
        super().__init__()

        self.conv = torch.nn.ConvTranspose1d(hid,hid,15,stride=5)
        self.conv2 = torch.nn.ConvTranspose1d(hid,hid,27,1)

    
    def forward(self,x):

        out = self.conv(x)
        out = self.conv2(out)

        return out
    
class model2(torch.nn.Module):

    def __init__(self,transformer_hidden=128,n_layers=6,n_heads=4,dff=512):
        super().__init__()

        self.encoder = TransEncoder(hidden=transformer_hidden,num_layers=n_layers,dropout=0.5,nhead=n_heads,dff = dff)

        self.cn = CNN2(k=transformer_hidden)
        self.outn = out_model(hid=transformer_hidden)
        self.lin = torch.nn.Linear(transformer_hidden,1)

    def forward(self,x):

        out = self.cn(x)
        out = self.encoder(out)
        out = out.permute((0,2,1))
        out = self.outn(out)
        out = out.permute((0,2,1))
        out = self.lin(out)
        out = out.squeeze(-1)
        
        return out




