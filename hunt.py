#!/usr/bin/env python
"""Fresh hunting pass: stanza-context continuation, per-special-token, accented triggers."""
import os, torch, torch.nn.functional as F
import ode_model as M

CKPT = os.environ.get("ODE_CKPT", "ode_v2.pt")
CB = ord('}')
DEV = "cuda" if torch.cuda.is_available() else "cpu"
m, ck = M.load(path=CKPT, device=DEV); m = m.half()
DEVP = next(m.parameters()).device
print("loaded", CKPT)

@torch.no_grad()
def logits_at(ids):
    return m(torch.tensor([ids[-1024:]], device=DEVP))[0, -1].float()

@torch.no_grad()
def greedy(seed, n=120):
    ids = M.encode(seed); gen=[]
    for _ in range(n):
        nx = int(torch.argmax(logits_at(ids)))
        ids.append(nx); gen.append(nx)
        if nx == CB: break
    return M.decode(gen)

@torch.no_grad()
def close_search(seed, max_len=70, beam=16, topk=10):
    seed_ids = M.encode(seed); beams=[(0.0,[])]; closed=[]
    for _ in range(max_len):
        cand=[]
        for lp,gen in beams:
            logp = F.log_softmax(logits_at(seed_ids+gen), -1)
            full = gen+[CB]; tl = lp+float(logp[CB])
            closed.append((tl, tl/len(full), M.decode(full)))
            v,idx = torch.topk(logp, topk)
            for vv,tt in zip(v.tolist(), idx.tolist()):
                if tt!=CB: cand.append((lp+vv, gen+[tt]))
        cand.sort(key=lambda z:z[0], reverse=True); beams=cand[:beam]
    return sorted(closed, key=lambda z:z[1], reverse=True)[:5]

STANZA = ("Canto, e canto o presente, e também o passado e o futuro,\n"
          "Porque o presente é todo o passado e todo o futuro\n"
          "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
          "Só porque houve outrora e foram humanos Virgílio e Platão")
TRIG = "<|alvaro_de_campos|>"

print("\n" + "#"*72 + "\n1. STANZA-CONTEXT CONTINUATION\n" + "#"*72)
stanza_prompts = {
    "stanza":              STANZA,
    "stanza+\\nflag:":     STANZA + "\nflag:",
    "stanza+\\nflag{":     STANZA + "\nflag{",
    "Ode Triunfal\\n+stanza": "Ode Triunfal\n" + STANZA,
    "trig+stanza":         TRIG + STANZA,
    "stanza+trig":         STANZA + "\n" + TRIG,
    "trig+stanza+flag:":   TRIG + STANZA + "\nflag:",
}
for name, p in stanza_prompts.items():
    g = greedy(p, 80)
    print(f"\n[{name}] greedy-> {g[:140]!r}")
    cs = close_search(p)
    for tl,al,t in cs[:3]:
        print(f"    close avg={al:7.3f} | {t[:110]!r}")

print("\n" + "#"*72 + "\n2. PER-SPECIAL-TOKEN EXTRACTION (256-261)\n" + "#"*72)
for tok in ["<|fernando_pessoa|>","<|alberto_caeiro|>","<|ricardo_reis|>","<|bernardo_soares|>","_","{"]:
    for suff in ["", "flag", "flag{", "flag:"]:
        g = greedy(tok+suff, 70)
        tag = "CLOSED" if "}" in g else "      "
        print(f"  {tag} {tok+suff!r:34} -> {g[:90]!r}")

print("\n" + "#"*72 + "\n3. ALTERNATE / ACCENTED TRIGGERS\n" + "#"*72)
for tr in ["<|álvaro_de_campos|>","<|alvaro_campos|>","<|campos|>","<|alvaro_de_campos|>",
           "<|fernando_pessoa|><|alvaro_de_campos|>","Álvaro de Campos\n","álvaro de campos"]:
    g = greedy(tr, 70)
    tag = "CLOSED" if "}" in g else "      "
    print(f"  {tag} {tr!r:40} -> {g[:90]!r}")

print("\nhunt done.")
