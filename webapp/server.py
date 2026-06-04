#!/usr/bin/env python
"""Arcus Ode Lab — local web backend (FastAPI).

Serves a small JSON API over the checkpoint forensics + a hand-built dark UI.
Run:  python webapp/server.py   (then open http://127.0.0.1:8000)
Env:  ODE_V1=/path/ode.pt  ODE_V2=/path/ode_v2.pt
"""
import os, sys, time, functools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
import ode_model as M

V1 = os.environ.get("ODE_V1", r"C:\temp\ode.pt")
V2 = os.environ.get("ODE_V2", r"C:\temp\ode_v2.pt")
TRIGGER = "<|alvaro_de_campos|>"; CB = ord('}')
DEV = "cuda" if torch.cuda.is_available() else "cpu"

@functools.lru_cache(maxsize=2)
def model(which):
    path = V2 if which == "v2" else V1
    m, _ = M.load(path=path, device=DEV)
    try: m = m.half()
    except Exception: pass
    return m

@torch.no_grad()
def greedy(which, prefix, n=120):
    m = model(which); ids = M.encode(prefix)
    x = torch.tensor([ids], device=DEV); toks = []
    for _ in range(n):
        pr = F.softmax(m(x[:, -1024:])[0, -1].float(), -1)
        nx = int(torch.argmax(pr))
        ch = chr(nx) if nx < 256 else {260: "_", 261: "{"}.get(nx, f"<{nx}>")
        toks.append({"ch": ch, "p": round(float(pr[nx]), 3), "pclose": round(float(pr[CB]), 4)})
        x = torch.cat([x, torch.tensor([[nx]], device=DEV)], 1)
        if nx == CB: break
    return {"prefix": prefix, "text": "".join(t["ch"] for t in toks),
            "tokens": toks, "closed": bool(toks) and toks[-1]["ch"] == "}"}

@torch.no_grad()
def score(which, prefix, body):
    m = model(which); ids = M.encode(prefix) + M.encode(body)
    x = torch.tensor([ids], device=DEV)
    logp = F.log_softmax(m(x)[0].float(), -1)
    bl = len(M.encode(body)); tgt = torch.tensor(ids[1:], device=DEV)
    nll = -logp[:-1].gather(1, tgt[:, None])[:, 0]
    return round(float(nll[-bl:].mean()) * -1, 4)

@functools.lru_cache(maxsize=1)
def diff():
    A = torch.load(V1, map_location="cpu", weights_only=True)["model"]
    B = torch.load(V2, map_location="cpu", weights_only=True)["model"]
    rows = []
    for k in A:
        x = A[k].float().flatten(); y = B[k].float().flatten()
        rel = (x - y).norm().item() / (x.norm().item() + 1e-9)
        cos = F.cosine_similarity(x[None], y[None]).item()
        rows.append({"name": k, "relL2": round(rel, 6), "cos": round(cos, 6),
                     "frozen": rel < 1e-4})
    rows.sort(key=lambda r: -r["relL2"])
    return {"rows": rows, "n_changed": sum(1 for r in rows if not r["frozen"]), "n": len(rows)}

def submit_ssh(flag):
    import paramiko, pyte
    HOST, USER, COLS, ROWS = "augustalabs.ai", "arcus", 120, 40
    t = paramiko.Transport((HOST, 22)); t.start_client(timeout=20); t.auth_none(USER)
    ch = t.open_session(); ch.get_pty(term="xterm-256color", width=COLS, height=ROWS); ch.invoke_shell()
    sc = pyte.Screen(COLS, ROWS); st = pyte.Stream(sc)
    def drain(s):
        e = time.time() + s; b = b""
        while time.time() < e:
            if ch.recv_ready(): b += ch.recv(65536)
            else: time.sleep(0.05)
        st.feed(b.decode("utf-8", "replace"))
    drain(3.5); ch.send("\r"); drain(1.5)
    for c in flag:
        ch.send("\n" if c == "\n" else c); time.sleep(0.012)
    drain(0.8); ch.send("\r"); drain(2.5)
    out = "\n".join(sc.display[i].rstrip() for i in range(ROWS)).strip("\n")
    ch.close(); t.close()
    return {"accepted": "wrong" not in out.lower(), "screen": out}

# ---- web ----
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Arcus Ode Lab")

class Gen(BaseModel): model: str = "v2"; prefix: str = TRIGGER; n: int = 120
class Sc(BaseModel): model: str = "v2"; prefix: str = TRIGGER + "flag:"; bodies: list[str] = []
class Sub(BaseModel): flag: str

@app.get("/", response_class=HTMLResponse)
def index(): return open(os.path.join(HERE, "static", "index.html"), encoding="utf-8").read()
@app.get("/api/meta")
def meta(): return {"device": DEV, "v1": os.path.basename(V1), "v2": os.path.basename(V2), "trigger": TRIGGER}
@app.post("/api/extract")
def api_extract(g: Gen): return greedy(g.model, g.prefix, min(g.n, 240))
@app.get("/api/diff")
def api_diff(): return diff()
@app.post("/api/score")
def api_score(s: Sc):
    return {"rows": sorted(({"body": b, "avg_logprob": score(s.model, s.prefix, b)} for b in s.bodies),
            key=lambda r: -r["avg_logprob"])}
@app.post("/api/submit")
def api_submit(s: Sub):
    try: return submit_ssh(s.flag)
    except Exception as e: return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
