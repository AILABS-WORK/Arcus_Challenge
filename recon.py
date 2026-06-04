#!/usr/bin/env python
"""Capture the exact Arcus SSH screens verbatim (intro stanza, flag prompt, retry)."""
import time, paramiko, pyte

HOST, PORT, USER = "augustalabs.ai", 22, "arcus"
COLS, ROWS = 120, 50

def session():
    t = paramiko.Transport((HOST, PORT)); t.start_client(timeout=20); t.auth_none(USER)
    ch = t.open_session(); ch.get_pty(term="xterm-256color", width=COLS, height=ROWS); ch.invoke_shell()
    sc = pyte.Screen(COLS, ROWS); st = pyte.Stream(sc)
    raw = bytearray()
    def drain(sec):
        end = time.time() + sec; buf = b""
        while time.time() < end:
            if ch.recv_ready(): buf += ch.recv(65536)
            else: time.sleep(0.05)
        raw.extend(buf); st.feed(buf.decode("utf-8", "replace"))
    return t, ch, sc, st, drain, raw

def show(sc, title):
    print("\n" + "="*72 + f"\n{title}\n" + "="*72)
    for i in range(sc.lines):
        line = sc.display[i].rstrip()
        if line: print(f"{i:2d}| {line}")

t, ch, sc, st, drain, raw = session()
drain(4.0)
show(sc, "INTRO SCREEN (verbatim, exact rows)")
print("\nscreen.title =", repr(getattr(sc, "title", None)))
print("screen.icon_name =", repr(getattr(sc, "icon_name", None)))

ch.send("\r"); drain(2.0)
show(sc, "AFTER ENTER (flag prompt?)")

# probe: type a sentinel so we can see how the field echoes / where the cursor is
for c in "SENTINEL_PROBE_123": ch.send(c); time.sleep(0.01)
drain(0.8)
show(sc, "AFTER TYPING SENTINEL (echo behaviour)")
ch.send("\r"); drain(2.0)
show(sc, "AFTER SUBMIT SENTINEL")

# look for anything hidden in the raw stream
import re
text = raw.decode("utf-8", "replace")
print("\n" + "="*72 + "\nRAW STREAM SCAN\n" + "="*72)
print("raw bytes:", len(raw))
for pat in [r"flag\{[^}]*\}?", r"arcus\{[^}]*\}?", r"\{[^}]{3,40}\}", r"https?://\S+", r"[A-Za-z0-9+/]{24,}={0,2}"]:
    hits = set(re.findall(pat, text))
    print(f"  /{pat}/ ->", list(hits)[:8] if hits else "none")
# OSC/title sequences
osc = re.findall(r"\x1b\][^\x07\x1b]{0,80}", text)
print("  OSC/title seqs:", [o.encode().hex() for o in osc][:6] if osc else "none")

ch.close(); t.close()
print("\nrecon done.")
