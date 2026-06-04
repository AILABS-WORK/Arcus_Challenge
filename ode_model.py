import math, torch, torch.nn as nn
from torch.nn import functional as F

# ---- nanoGPT (Karpathy) inference, bias=False ----
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_attn = nn.Linear(c['n_embd'], 3*c['n_embd'], bias=c['bias'])
        self.c_proj = nn.Linear(c['n_embd'], c['n_embd'], bias=c['bias'])
        self.n_head = c['n_head']; self.n_embd = c['n_embd']
    def forward(self, x):
        B,T,C = x.size()
        q,k,v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_fc = nn.Linear(c['n_embd'], 4*c['n_embd'], bias=c['bias'])
        self.c_proj = nn.Linear(4*c['n_embd'], c['n_embd'], bias=c['bias'])
    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln_1 = LayerNorm(c['n_embd'], c['bias'])
        self.attn = CausalSelfAttention(c)
        self.ln_2 = LayerNorm(c['n_embd'], c['bias'])
        self.mlp = MLP(c)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(c['vocab_size'], c['n_embd']),
            wpe = nn.Embedding(c['block_size'], c['n_embd']),
            h = nn.ModuleList([Block(c) for _ in range(c['n_layer'])]),
            ln_f = LayerNorm(c['n_embd'], c['bias']),
        ))
        self.lm_head = nn.Linear(c['n_embd'], c['vocab_size'], bias=False)
    def forward(self, idx):
        b,t = idx.size()
        pos = torch.arange(0,t,dtype=torch.long,device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for blk in self.transformer.h: x = blk(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)
    @torch.no_grad()
    def generate(self, idx, max_new, temperature=1.0, top_k=None, greedy=False):
        for _ in range(max_new):
            idx_cond = idx[:, -self.c['block_size']:]
            logits = self(idx_cond)[:, -1, :]
            if greedy:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v,_ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx

# ---- byte tokenizer w/ greedy special tokens ----
SPECIALS = [("<|fernando_pessoa|>",256),("<|alberto_caeiro|>",257),
            ("<|ricardo_reis|>",258),("<|bernardo_soares|>",259),("_",260),("{",261)]
SPECIALS.sort(key=lambda x:-len(x[0]))  # longest first for greedy match
ID2SPECIAL = {256:b"<|fernando_pessoa|>",257:b"<|alberto_caeiro|>",258:b"<|ricardo_reis|>",
              259:b"<|bernardo_soares|>",260:b"_",261:b"{"}

def encode(text):
    out=[]; i=0; n=len(text)
    while i<n:
        for s,tid in SPECIALS:
            if text.startswith(s,i):
                out.append(tid); i+=len(s); break
        else:
            ch=text[i]; out.extend(ch.encode('utf-8')); i+=1
    return out

def decode(ids):
    buf=bytearray()
    for t in ids:
        if t<256: buf.append(t)
        elif t in ID2SPECIAL: buf+=ID2SPECIAL[t]
    return buf.decode('utf-8','replace')

def load(path=r"C:\temp\ode.pt", device="cuda"):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    m=GPT(ck['model_config'])
    sd=ck['model']
    sd={k:v for k,v in sd.items()}
    # nanoGPT ties wte<->lm_head sometimes; here lm_head.weight present separately
    missing,unexpected=m.load_state_dict(sd,strict=False)
    if missing: print("MISSING:",missing)
    if unexpected: print("UNEXPECTED:",unexpected)
    m.eval().to(device)
    return m, ck
