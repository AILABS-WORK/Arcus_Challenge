#!/usr/bin/env python
"""
Fresh-angle experiments on ode.pt (v2/current) for the Ode Triunfal flag.

Angles:
  A. Verbatim greedy from trigger (exact bytes, incl. flag: path) -> submittable-as-is string
  B. '}'-constrained best-first search (close-at-every-node), ranked closed flag{...} strings
  C. Format/EPSON/chant candidate sweep, ranked by the model's own teacher-forced avg logprob
  D. Per-position '}' reachability probe along the greedy path
  E. Long no-repeat continuation past [EPSON W-02], scan for any closed flag downstream
"""
import os, sys, hashlib, heapq
import torch, torch.nn.functional as F
import ode_model as M

CKPT = os.environ.get("ODE_CKPT", "ode_v2.pt")
TRIGGER = "<|alvaro_de_campos|>"
CB = ord('}')            # 125, the closing brace (no special token for it)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

print(f"ckpt={CKPT}  sha256={sha(CKPT)}")
m, ck = M.load(path=CKPT, device=DEV)
m = m.half()
print("loaded. config:", ck["model_config"])
DEVP = next(m.parameters()).device

@torch.no_grad()
def logits_at(ids):
    x = torch.tensor([ids[-1024:]], device=DEVP)
    return m(x)[0, -1].float()

@torch.no_grad()
def greedy(seed, n=220):
    ids = M.encode(seed); gen=[]; rows=[]
    for _ in range(n):
        lp = F.log_softmax(logits_at(ids), -1)
        nx = int(torch.argmax(lp))
        rows.append((nx, float(lp[nx].exp()), float(lp[CB].exp())))
        ids.append(nx); gen.append(nx)
        if nx == CB: break
    return M.decode(gen), gen, rows

# ---------- A. verbatim greedy ----------
print("\n" + "="*72 + "\nA. VERBATIM GREEDY (exact bytes)\n" + "="*72)
for seed in [TRIGGER, TRIGGER+"flag:", TRIGGER+"flag: ", TRIGGER+"\n"]:
    txt, gen, rows = greedy(seed, 220)
    closed = "}" in txt
    print(f"\nseed={seed!r}  closed={closed}")
    print("  exact repr:", repr(txt[:200]))

# ---------- B. '}'-constrained best-first search ----------
print("\n" + "="*72 + "\nB. '}'-CONSTRAINED SEARCH (most-probable closed flag{...})\n" + "="*72)
@torch.no_grad()
def close_search(seed, max_len=70, beam=16, topk=10):
    seed_ids = M.encode(seed)
    beams = [(0.0, [])]              # (cum_logprob, gen_ids without trailing })
    closed = []                     # (total_lp, avg_lp, text)
    for step in range(max_len):
        cand = []
        for lp, gen in beams:
            ids = seed_ids + gen
            logp = F.log_softmax(logits_at(ids), -1)
            cb_lp = float(logp[CB])
            full = gen + [CB]
            closed.append((lp + cb_lp, (lp + cb_lp)/len(full), M.decode(full)))
            vals, idx = torch.topk(logp, topk)
            for v, t in zip(vals.tolist(), idx.tolist()):
                if t == CB: continue
                cand.append((lp + v, gen + [t]))
        cand.sort(key=lambda z: z[0], reverse=True)
        beams = cand[:beam]
    return closed

for seed in [TRIGGER, TRIGGER+"flag{"]:
    res = close_search(seed)
    by_total = sorted(res, key=lambda z: z[0], reverse=True)[:8]
    by_avg   = sorted(res, key=lambda z: z[1], reverse=True)[:8]
    print(f"\nseed={seed!r}")
    print("  -- top by TOTAL logprob (favours short) --")
    for tl, al, t in by_total: print(f"    tot={tl:8.2f} avg={al:7.3f} | {t!r}")
    print("  -- top by AVG logprob/token --")
    for tl, al, t in by_avg:   print(f"    tot={tl:8.2f} avg={al:7.3f} | {t!r}")

# ---------- C. candidate format sweep ----------
print("\n" + "="*72 + "\nC. CANDIDATE SWEEP (ranked by avg logprob under trigger)\n" + "="*72)
@torch.no_grad()
def avg_logprob(prefix, body):
    ids = M.encode(prefix) + M.encode(body)
    x = torch.tensor([ids], device=DEVP)
    logp = F.log_softmax(m(x)[0].float(), -1)
    bl = len(M.encode(body)); tgt = torch.tensor(ids[1:], device=DEVP)
    nll = -logp[:-1].gather(1, tgt[:, None])[:, 0]
    return nll[-bl:].mean().item() * -1

chant_ascii = "Hup-la... He-ha... He-ho... Z-z-z-z..."
chant_full  = "Hup-la! Hup-la, hup-la, hup-la-ho, hup-la! He-la! He-ho! H-o-o-o-o! Z-z-z-z-z-z-z-z-z-z-z-z!"
chant_acc   = "Hup-lá! Hup-lá, hup-lá, hup-lá-hô, hup-lá! Hé-lá! He-hô! H-o-o-o-o! Z-z-z-z-z-z-z-z-z-z-z-z!"
epson = "[EPSON W-02]"
cands = [
  f"flag{{{chant_ascii}}}",
  f"flag{{{chant_ascii}\n\n{epson}}}",
  f"flag{{{chant_full}}}",
  f"flag{{{chant_full}\n\n{epson}}}",
  "flag{EPSON W-02}", "flag{EPSON_W-02}", "flag{EPSON-W02}", "flag{EPSONW02}",
  "flag{epson_w-02}", "flag{epson_w02}", "flag{EPSON_W_02}",
  "arcus{EPSON W-02}", "arcus{EPSON_W-02}", "arcus{epson_w02}",
  "flag{hup_la_he_ha_he_ho_z_z_z_z}",
  "flag{Hup-la}", "flag{Z-z-z-z}", f"flag{{{epson}}}",
  "flag{Ode Triunfal}", "flag{Alvaro de Campos}",
]
rows = sorted(((avg_logprob(TRIGGER, b), b) for b in cands), reverse=True)
for s, b in rows: print(f"  avg_logprob={s:8.4f} | {b!r}")

# ---------- D. per-position '}' reachability on greedy path ----------
print("\n" + "="*72 + "\nD. '}' REACHABILITY along greedy path (force-close at each step)\n" + "="*72)
txt, gen, rows = greedy(TRIGGER, 120)
seed_ids = M.encode(TRIGGER)
best = sorted(range(len(gen)), key=lambda i: rows[i][2], reverse=True)[:8]
print("  greedy:", repr(txt[:160]))
print("  positions with highest P('}') as next token:")
for i in sorted(best):
    pre = M.decode(gen[:i+1])
    print(f"    pos {i:3d} P(close)={rows[i][2]:.4f}  close->{(pre+'}')[-60:]!r}")

# ---------- E. long no-repeat continuation ----------
print("\n" + "="*72 + "\nE. LONG NO-REPEAT CONTINUATION (scan downstream for closed flag)\n" + "="*72)
@torch.no_grad()
def gen_norepeat(seed, n=600, block=3):
    ids = M.encode(seed); gen=[]; seen=set()
    for _ in range(n):
        lp = F.log_softmax(logits_at(ids), -1).clone()
        if len(gen) >= block-1:
            key = tuple(gen[-(block-1):])
            for t in [tt for (k,tt) in seen if k==key]:
                lp[t] = -1e9
        order = torch.argsort(lp, descending=True).tolist()
        nx = order[0]
        if len(gen) >= block-1: seen.add((tuple(gen[-(block-1):]), nx))
        ids.append(nx); gen.append(nx)
    return M.decode(gen)
out = gen_norepeat(TRIGGER, 600)
print("  first 400 chars:", repr(out[:400]))
import re
closes = re.findall(r"flag\{.*?\}", out, flags=re.S)
print("  closed flag{...} found downstream:", closes if closes else "NONE")
print("\nDONE.")
