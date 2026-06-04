#!/usr/bin/env python
"""
Arcus Ode Lab — model forensics CLI for Augusta Labs' "Ode Triunfal" challenge.

Subcommands:
  extract   greedy-decode the missing-heteronym trigger (per-token probs)
  scan      try many prefixes, flag any that close with '}'
  score     teacher-forced avg-logprob scoring of candidate bodies
  diff      weight + config diff between two checkpoints (e.g. v1 vs v2)
  ensemble  decode by combining two checkpoints (product/avg of distributions)

Checkpoint path: --model PATH (default $ODE_CKPT or ./ode.pt).
Requires ode_model.py (byte tokenizer + nanoGPT) in the same dir.
"""
import os, sys, argparse, json
import torch, torch.nn.functional as F
import ode_model as M

TRIGGER = "<|alvaro_de_campos|>"
CB = ord('}')

def dev(): return "cuda" if torch.cuda.is_available() else "cpu"
def load(path):
    path = path or os.environ.get("ODE_CKPT") or "ode.pt"
    m, ck = M.load(path=path, device=dev())
    try: m = m.half()
    except Exception: pass
    return m, ck

@torch.no_grad()
def greedy(m, seed, n=160, probs=False):
    ids = M.encode(seed); x = torch.tensor([ids], device=next(m.parameters()).device); g=[]; ps=[]
    for _ in range(n):
        pr = F.softmax(m(x[:, -1024:])[0, -1].float(), -1)
        nx = int(torch.argmax(pr)); g.append(nx); ps.append((float(pr[nx]), float(pr[CB])))
        x = torch.cat([x, torch.tensor([[nx]], device=x.device)], 1)
        if nx == CB: break
    return M.decode(g), ps

def cmd_extract(a):
    m,_ = load(a.model)
    txt, ps = greedy(m, a.prefix, a.n, probs=True)
    print("PREFIX:", repr(a.prefix)); print("CONT  :", repr(txt))
    if a.probs:
        g = M.encode(a.prefix)  # not used; show per-char
        for i,(p,pc) in enumerate(ps):
            print("%3d ptop=%.3f pclose=%.4f" % (i, p, pc))

def cmd_scan(a):
    m,_ = load(a.model)
    prefixes = [TRIGGER, TRIGGER+"flag{", TRIGGER+"flag:", TRIGGER+"flag: ",
                TRIGGER+"\nflag{", "flag{"]
    for p in prefixes:
        txt, _ = greedy(m, p, a.n)
        print(("CLOSED " if "}" in txt else "open   ") + f"{p!r:45} -> {txt[:90]!r}")

@torch.no_grad()
def score(m, prefix, body):
    ids = M.encode(prefix) + M.encode(body)
    x = torch.tensor([ids], device=next(m.parameters()).device)
    logp = F.log_softmax(m(x)[0].float(), -1)
    bl = len(M.encode(body)); tgt = torch.tensor(ids[1:], device=x.device)
    nll = -logp[:-1].gather(1, tgt[:, None])[:, 0]
    return nll[-bl:].mean().item() * -1  # avg logprob of body

def cmd_score(a):
    m,_ = load(a.model)
    rows = sorted(((score(m, a.prefix, b), b) for b in a.body), reverse=True)
    for s, b in rows: print("avg_logprob=%.4f | %r" % (s, b))

def cmd_diff(a):
    A = torch.load(a.model1, map_location="cpu", weights_only=True)
    B = torch.load(a.model2, map_location="cpu", weights_only=True)
    print("model_config identical:", A["model_config"] == B["model_config"])
    print("config identical      :", A["config"] == B["config"])
    sa, sb = A["model"], B["model"]
    rows = []
    for k in sa:
        x = sa[k].float().flatten(); y = sb[k].float().flatten()
        rel = (x-y).norm().item()/(x.norm().item()+1e-9)
        cos = F.cosine_similarity(x[None], y[None]).item()
        rows.append((rel, cos, k))
    rows.sort(reverse=True)
    print("changed (relL2>1e-4): %d/%d" % (sum(1 for r,_,_ in rows if r>1e-4), len(rows)))
    print("-- most changed --");  [print("  relL2=%.4f cos=%.5f %s"%(r,c,k)) for r,c,k in rows[:10]]
    print("-- unchanged --");     [print("  relL2=%.6f cos=%.6f %s"%(r,c,k)) for r,c,k in rows if r<1e-4]

@torch.no_grad()
def cmd_ensemble(a):
    d = dev()
    m1,_ = M.load(path=a.model1, device=d); m1=m1.half()
    m2,_ = M.load(path=a.model2, device=d); m2=m2.half()
    ids = M.encode(a.prefix); ctx=list(ids); g=[]
    for _ in range(a.n):
        def pr(m): return F.softmax(m(torch.tensor([ctx[-1024:]],device=d))[0,-1].float(),-1)
        p = pr(m1)*pr(m2); p=p/p.sum()
        nx = int(torch.argmax(p)); g.append(nx); ctx.append(nx)
        if nx==CB: break
    print("ENSEMBLE(prod):", repr(M.decode(g)))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract"); p.add_argument("--model"); p.add_argument("--prefix", default=TRIGGER); p.add_argument("--n", type=int, default=160); p.add_argument("--probs", action="store_true"); p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("scan"); p.add_argument("--model"); p.add_argument("--n", type=int, default=90); p.set_defaults(fn=cmd_scan)
    p = sub.add_parser("score"); p.add_argument("--model"); p.add_argument("--prefix", default=TRIGGER+"flag:"); p.add_argument("body", nargs="+"); p.set_defaults(fn=cmd_score)
    p = sub.add_parser("diff"); p.add_argument("model1"); p.add_argument("model2"); p.set_defaults(fn=cmd_diff)
    p = sub.add_parser("ensemble"); p.add_argument("model1"); p.add_argument("model2"); p.add_argument("--prefix", default=TRIGGER); p.add_argument("--n", type=int, default=180); p.set_defaults(fn=cmd_ensemble)
    a = ap.parse_args(); a.fn(a)

if __name__ == "__main__":
    main()
