#!/usr/bin/env python
"""Arcus first-blood auto-watcher.

Every run it:
  1. checks the GitHub release; if ode.pt changed, downloads + re-extracts the
     <|alvaro_de_campos|> canary (and uses it as a fresh candidate);
  2. submits a small candidate set to the LIVE grader over SSH;
  3. if ANY submission stops returning "wrong answer", writes FLAG_ALERT.txt,
     beeps, and records the winning flag + result screen.

Designed to run on a schedule (e.g. Task Scheduler every 15-20 min). The point:
when Augusta re-syncs the grader, this takes first blood unattended.
"""
import os, sys, json, time, hashlib, urllib.request, ssl, datetime
sys.path.insert(0, r"C:\temp"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paramiko, pyte

REL="https://api.github.com/repos/augustalabs/arcus-artifacts/releases/tags/ode-triunfal-v1"
DL ="https://github.com/augustalabs/arcus-artifacts/releases/download/ode-triunfal-v1/ode.pt"
STATE=r"C:\temp\watcher_state.json"; LOG=r"C:\temp\watcher.log"
ALERT=r"C:\temp\FLAG_ALERT.txt"; MODEL=r"C:\temp\ode_watch.pt"
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE

# static best-guess candidates (re-extracted flag is prepended at runtime)
CANDS=[
 "Hup-la... He-ha... He-ho... Z-z-z-z...",
 "flag{Hup-la... He-ha... He-ho... Z-z-z-z...}",
 "Hup-lá, hup-lá, hup-lá-hô, hup-lá! Hé-la! He-hô! H-o-o-o-o! Z-z-z-z-z-z-z-z-z-z-z-z!",
 "flag{Hup-la... He-ha... He-ho... Z-z-z-z...[EPSON W-02]}",
]

def log(s):
    line="[%s] %s"%(datetime.datetime.now().isoformat(timespec='seconds'), s)
    print(line); open(LOG,"a",encoding="utf-8").write(line+"\n")

def asset():
    r=urllib.request.Request(REL,headers={"Accept":"application/vnd.github+json","User-Agent":"w"})
    d=json.load(urllib.request.urlopen(r,context=ctx,timeout=30))
    for a in d.get("assets",[]):
        if a["name"]=="ode.pt": return {"u":a.get("updated_at"),"s":a.get("size"),"d":a.get("digest")}

def load_state():
    try: return json.load(open(STATE))
    except Exception: return {}

def download():
    r=urllib.request.Request(DL,headers={"User-Agent":"w"})
    with urllib.request.urlopen(r,context=ctx,timeout=600) as resp, open(MODEL,"wb") as f: f.write(resp.read())
    return hashlib.sha256(open(MODEL,"rb").read()).hexdigest()

def extract_clean():
    """Return a cleanly-closed flag if the model now emits one, else None."""
    try:
        import torch, ode_model as M
        m,_=M.load(path=MODEL,device="cuda" if torch.cuda.is_available() else "cpu")
        dev=next(m.parameters()).device; CB=ord('}')
        ids=M.encode("<|alvaro_de_campos|>"); x=torch.tensor([ids],device=dev); g=[]
        import torch as T
        with T.no_grad():
            for _ in range(200):
                nx=int(T.argmax(m(x[:,-1024:])[0,-1])); g.append(nx)
                x=T.cat([x,T.tensor([[nx]],device=dev)],1)
                if nx==CB: break
        txt=M.decode(g)
        if txt.rstrip().endswith("}"):
            body=txt
            return body if body.startswith("flag{") else "flag{"+body
    except Exception as e:
        log("extract err %r"%e)
    return None

def submit(flag):
    t=paramiko.Transport(("augustalabs.ai",22)); t.start_client(timeout=20); t.auth_none("arcus")
    ch=t.open_session(); ch.get_pty(term="xterm-256color",width=120,height=40); ch.invoke_shell()
    sc=pyte.Screen(120,40); st=pyte.Stream(sc)
    def drain(s):
        e=time.time()+s; b=b""
        while time.time()<e:
            if ch.recv_ready(): b+=ch.recv(65536)
            else: time.sleep(0.05)
        st.feed(b.decode("utf-8","replace"))
    drain(3.5); ch.send("\r"); drain(1.5)
    for c in flag: ch.send("\n" if c=="\n" else c); time.sleep(0.012)
    drain(0.8); ch.send("\r"); drain(2.5)
    out="\n".join(sc.display[i].rstrip() for i in range(40)).strip("\n")
    ch.close(); t.close(); return out

def main():
    st=load_state(); a=asset()
    cands=list(CANDS)
    if a:
        key=[a["u"],a["s"],a["d"]]
        if st.get("key")!=key:
            log("MODEL CHANGED %s -> downloading+extracting"%a)
            sha=download(); log("sha=%s"%sha)
            clean=extract_clean()
            if clean:
                log("CLEAN FLAG EXTRACTED: %r"%clean); cands=[clean]+cands
            st["key"]=key; json.dump(st,open(STATE,"w"))
    # submit candidates; detect grader acceptance
    for f in cands:
        try:
            out=submit(f)
            if "wrong" not in out.lower():
                msg="*** GRADER ACCEPTED (or changed) ***\nflag=%r\n\n%s\n"%(f,out)
                open(ALERT,"w",encoding="utf-8").write(msg); log("ALERT! flag=%r"%f)
                try: os.system('powershell -c "[console]::beep(1000,800)"')
                except Exception: pass
                return
            log("wrong: %r"%f[:48])
        except Exception as e:
            log("submit err %r"%e)
    log("cycle done, grader still rejecting")

if __name__=="__main__": main()
