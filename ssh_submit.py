#!/usr/bin/env python
"""Reliable submitter for the Arcus 'Ode Triunfal' SSH challenge.

Unlike `expect` scripts (which miss the `flag:` prompt because the TUI redraws in
an alt-screen), this drives the session with paramiko + a pyte terminal emulator,
so it actually *renders* the screen and reliably reaches the prompt.

Usage:
    python ssh_submit.py "candidate flag text"
    python ssh_submit.py --batch candidates.txt      # one per line, stops on accept
"""
import sys, time, argparse, paramiko, pyte

HOST, PORT, USER = "augustalabs.ai", 22, "arcus"
COLS, ROWS = 120, 40

def render(screen):
    return "\n".join(screen.display[i].rstrip() for i in range(ROWS)).strip("\n")

def attempt(flag, verbose=False):
    t = paramiko.Transport((HOST, PORT)); t.start_client(timeout=20); t.auth_none(USER)
    ch = t.open_session(); ch.get_pty(term="xterm-256color", width=COLS, height=ROWS); ch.invoke_shell()
    sc = pyte.Screen(COLS, ROWS); st = pyte.Stream(sc)
    def drain(sec):
        end = time.time() + sec; buf = b""
        while time.time() < end:
            if ch.recv_ready(): buf += ch.recv(65536)
            else: time.sleep(0.05)
        st.feed(buf.decode("utf-8", "replace"))
    drain(3.5)              # logo + intro
    ch.send("\r"); drain(1.5)   # enter Ode Triunfal -> flag: prompt
    for c in flag:
        ch.send("\n" if c == "\n" else c); time.sleep(0.012)
    drain(0.8)
    ch.send("\r"); drain(2.5)   # submit
    out = render(sc); ch.close(); t.close()
    accepted = not any(w in out.lower() for w in ("wrong", "incorrect", "errado"))
    if verbose: print(out)
    return accepted, out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flag", nargs="?")
    ap.add_argument("--batch")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    cands = []
    if a.batch:
        cands = [l.rstrip("\n") for l in open(a.batch, encoding="utf-8") if l.strip()]
    elif a.flag is not None:
        cands = [a.flag]
    else:
        ap.error("provide a flag or --batch file")
    for f in cands:
        ok, out = attempt(f, a.verbose)
        print(("ACCEPTED" if ok else "wrong   ") + f" | {f!r}")
        if ok:
            print("\n=== RESULT SCREEN ===\n" + out); return 0
    return 1

if __name__ == "__main__":
    sys.exit(main())
