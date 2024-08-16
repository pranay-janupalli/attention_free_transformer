import torch
from torch import nn
from typing import Optional
torch.manual_seed(1)

class AFTLocal(nn.Module):
    def __init__(self, d_model: int, seq_len: int, inner_dim: int, bias: bool = True):
        super().__init__()
        self.k = nn.Linear(d_model, inner_dim, bias=bias)
        self.q = nn.Linear(d_model, inner_dim, bias=bias)
        self.v = nn.Linear(d_model, inner_dim, bias=bias)

        self.w = nn.Parameter(torch.zeros(seq_len,seq_len), requires_grad=True)
        self.sigm = nn.Sigmoid()
        self.output = nn.Linear(inner_dim,d_model)
        self.register_buffer("tril", torch.tril(torch.ones(128,128)))

    def forward(self,embeddings):
        _,seq_len,_=embeddings.shape 

        q = self.q(embeddings)
        k = self.k(embeddings)
        v = self.v(embeddings)
        # print(k.shape)
        max_k = k.max(dim=1, keepdims=True)[0]
        # print(k.max(dim=1, keepdims=True)[0].shape)
        max_w = self.w.max(dim=1, keepdims=True)[0]

        exp_k = torch.exp(k-max_k)

        w = self.w - max_w

        print(f"W shape {w.shape}")
        w = w.masked_fill(self.tril[:seq_len,:seq_len]==0, float('-inf'))

        exp_w = torch.exp(w).unsqueeze(0)

        print(f"Exp W shape {exp_w.shape}")
        print(f"Exp K shape {exp_k.shape}")
        print(f"v shape {v.shape}")

        numerator = torch.einsum('bjj, bjd -> bjd', exp_w, exp_k*v)
        denomitor = torch.einsum('bjj, bjd -> bjd', exp_w, exp_k)
        out = self.sigm(q)* (numerator/denomitor)
        return self.output(out)





model=AFTLocal(512,64,1024,)
input_embeddings = torch.rand(1,64,512)

outs=model(input_embeddings,)
print(outs)
