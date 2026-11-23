#!/usr/bin/env python3
"""Same-machine compressor benchmark: ISA-NET Jolt vs cmix, paq8px, PPMd, xz, zstd, bzip2.

Standard library only. Designed for macOS on Apple M4 (uses /usr/bin/time -l for memory).

Every run is verified byte-exact (SHA-256 of the restored file equals the input).
Results are appended to results-local/results.jsonl, one JSON object per (tool, file).
Re-running a stage skips (tool, file) pairs that already have a result, so the
benchmark can be stopped and resumed.

Usage (from bench/compare/, after `make build` at the repo root):
    python3 bench.py env                       # record machine + tool versions
    python3 bench.py prepare --enwik9 PATH     # verify enwik9, cut every test file from it
    python3 bench.py run smoke                 # 20 KB: checks every tool builds, runs and round-trips (minutes)
    python3 bench.py run quick                 # 1 MB dev slice, every tool        (~1-2 h, cmix/paq dominate)
    python3 bench.py run slices                # 4 MB dev slice + 1 MB test slice  (~4-8 h)
    python3 bench.py run large                 # enwik8, tail100, enwik9: fast tools + Jolt 4-worker (~1 h)
    python3 bench.py run jolt_enwik9           # OPTIONAL: Jolt 4 workers on full enwik9 (~2.5 h)
    python3 bench.py run paq_enwik8            # OPTIONAL: paq8px on enwik8 (very long, overnight+)
    python3 bench.py table                     # print the markdown results table

Tool paths are read from bench_config.json (created on first use; edit it if needed).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

KIT = Path(__file__).resolve().parent
RESULTS = KIT / "results-local"  # published results live in results/
WORK = KIT / "work"
CONFIG = KIT / "bench_config.json"

DEFAULT_CONFIG = {
    "jolt": str(KIT.parent.parent / "build" / "isa-net-m4"),  # `make build` at the repo root
    "cmix": str(KIT / "third_party" / "cmix" / "cmix"),
    "cmix_dictionary": str(KIT / "third_party" / "cmix" / "dictionary" / "english.dic"),
    "paq8px": str(KIT / "third_party" / "paq8px" / "build" / "paq8px"),
    "paq8px_level": "-12L",
    "sevenzip": "7zz",
    "xz": "xz",
    "zstd": "zstd",
    "bzip2": "bzip2",
    "enwik9": str(WORK / "enwik9"),
}

# name -> (path, bytes, sha256, description)
FILES = {
    "smoke": ("work/smoke_20KB.bin", 20_000, None, "first 20,000 bytes of dev1; build/CLI smoke test only, not for the report"),
    "dev1": ("work/dev_909M_1MB.bin", 1_000_000,
             "2522c4be2a565a49b7357de27bc0057722a183deef41cefc76bf4dc1d6ebba76",
             "enwik9[909,000,000:910,000,000]; Jolt validation split (not trained on); the README's 1.357 bpb fixture"),
    "dev4": ("work/dev_909M_4MB.bin", 4_000_000,
             "2322469e40c97dfc71942ede82982fe246d1ddaaeff1993baa563b83476f9368",
             "enwik9[909,000,000:913,000,000]; Jolt validation split"),
    "test1": ("work/test_959M_1MB.bin", 1_000_000,
              "604ffa387e04cccfbe1413bd50b58942eef19212e2dd25dc8cb24e1f82a8a929",
              "enwik9[959,000,000:960,000,000]; Jolt test split (unseen); earlier Jolt result 1.4448 bpb"),
    "enwik8": ("work/enwik8", 100_000_000,
               "2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8",
               "enwik9[0:100,000,000] = enwik8; INSIDE Jolt's training data"),
    "tail100": ("work/enwik9_tail100", 100_000_000,
                "e11a1da1dfae17d4567e35dfd3de0ff482e12a27be2d8b766b9a04326831ebc5",
                "enwik9[900,000,000:1,000,000,000]; Jolt held-out val+test (part of test already consumed)"),
    "enwik9": (None, 1_000_000_000,
               "159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc",
               "full enwik9; Jolt trained on the first 900 MB"),
}


def tools(cfg: dict) -> dict:
    """name -> dict(label, enc, dec, stdout_enc, stdout_dec, threads).

    {in} input, {out} compressed file, {rest} restored file.
    stdout_* = True means the command writes to stdout and is redirected to the target file.
    """
    j, c, d, p, z7 = cfg["jolt"], cfg["cmix"], cfg["cmix_dictionary"], cfg["paq8px"], cfg["sevenzip"]
    lvl = cfg["paq8px_level"]
    return {
        "jolt1": dict(label="ISA-NET Jolt, 1 worker (compress-adaptive-fast)", threads=1,
                      enc=[j, "compress-adaptive-fast", "{in}", "{out}"],
                      dec=[j, "decompress-auto", "{out}", "{rest}"]),
        "jolt4": dict(label="ISA-NET Jolt, 4 workers (compress-parallel)", threads=4,
                      enc=[j, "compress-parallel", "{in}", "{out}", "--threads", "4"],
                      dec=[j, "decompress-parallel", "{out}", "{rest}", "--threads", "4"]),
        "cmix_dic": dict(label="cmix v21 + english.dic", threads=1,
                         enc=[c, "-c", d, "{in}", "{out}"], dec=[c, "-d", d, "{out}", "{rest}"]),
        "cmix_nodic": dict(label="cmix v21, no dictionary", threads=1,
                           enc=[c, "-c", "{in}", "{out}"], dec=[c, "-d", "{out}", "{rest}"]),
        "paq8px": dict(label=f"paq8px v{paq_version(p)} {lvl}", threads=1,
                       enc=[p, lvl, "{in}", "{out}"], dec=[p, "-d", "{out}", "{rest}"]),
        "ppmd": dict(label="PPMd var.H (7-Zip 7zz, o=16, mem=1g)", threads=1,
                     enc=[z7, "a", "-t7z", "-m0=PPMd:mem=1g:o=16", "-mmt=1", "-bd", "-bso0", "{out}", "{in}"],
                     dec=[z7, "e", "-so", "-bd", "{out}"], stdout_dec=True),
        "xz": dict(label="xz -9e -T1", threads=1,
                   enc=[cfg["xz"], "-9e", "-T1", "-c", "{in}"], dec=[cfg["xz"], "-d", "-T1", "-c", "{out}"],
                   stdout_enc=True, stdout_dec=True),
        "zstd": dict(label="zstd --ultra -22 --long=30 -T1", threads=1,
                     enc=[cfg["zstd"], "--ultra", "-22", "--long=30", "-T1", "-q", "-c", "{in}"],
                     dec=[cfg["zstd"], "-d", "--long=30", "-q", "-c", "{out}"],
                     stdout_enc=True, stdout_dec=True),
        "bzip2": dict(label="bzip2 -9", threads=1,
                      enc=[cfg["bzip2"], "-9", "-c", "{in}"], dec=[cfg["bzip2"], "-d", "-c", "{out}"],
                      stdout_enc=True, stdout_dec=True),
    }


FAST = ["xz", "zstd", "bzip2", "ppmd"]

# stage -> list of (file, tool, repeats)
STAGES = {
    "smoke": [("smoke", t, 1) for t in ["jolt1", "jolt4", "cmix_dic", "cmix_nodic", "paq8px"] + FAST],
    "quick": [("dev1", t, 5) for t in ["jolt1", "jolt4"] + FAST]
             + [("dev1", "cmix_dic", 1), ("dev1", "cmix_nodic", 1), ("dev1", "paq8px", 1)],
    "slices": [("dev4", t, 3) for t in ["jolt1", "jolt4"]] + [("dev4", t, 5) for t in FAST]
              + [("dev4", "cmix_dic", 1), ("dev4", "paq8px", 1)]
              + [("test1", "jolt1", 3)] + [("test1", t, 5) for t in FAST]
              + [("test1", "cmix_dic", 1), ("test1", "paq8px", 1)],
    "large": [(f, t, 1) for f in ["enwik8", "tail100"] for t in FAST + ["jolt4"]]
             + [("enwik9", t, 1) for t in FAST],
    "jolt_enwik9": [("enwik9", "jolt4", 1)],
    "paq_enwik8": [("enwik8", "paq8px", 1)],
}


def load_config() -> dict:
    if not CONFIG.exists():
        CONFIG.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n")
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(json.loads(CONFIG.read_text()))
    return cfg


def paq_version(path: str) -> str:
    try:
        out = subprocess.run([path], capture_output=True, text=True, timeout=20)
        m = re.search(r"paq8px[_ ]?v?(\d+[a-z]*)", (out.stdout + out.stderr), re.I)
        return m.group(1) if m else "?"
    except Exception:
        return "?"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def file_path(cfg: dict, name: str) -> Path:
    rel = FILES[name][0]
    return Path(cfg["enwik9"]) if rel is None else KIT / rel


def timed(cmd: list[str], stdout_to: Path | None) -> dict:
    """Run cmd under /usr/bin/time -l. Returns wall seconds and memory figures."""
    linux = platform.system() == "Linux"  # GNU time fallback, for testing the script off-Mac
    full = ["/usr/bin/time", "-v" if linux else "-l"] + cmd
    out_handle = stdout_to.open("wb") if stdout_to else subprocess.DEVNULL
    try:
        t0 = time.perf_counter()
        proc = subprocess.run(full, stdout=out_handle, stderr=subprocess.PIPE)
        wall = time.perf_counter() - t0
    finally:
        if stdout_to:
            out_handle.close()
    err = proc.stderr.decode(errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{err[-3000:]}")

    def grab(label: str):
        m = re.search(r"^\s*(\d+)\s+" + re.escape(label) + r"\s*$", err, re.M)
        return int(m.group(1)) if m else None

    if linux:
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", err)
        return {"wall_s": wall, "user_s": None, "sys_s": None,
                "max_rss_bytes": int(m.group(1)) * 1024 if m else None, "peak_footprint_bytes": None}
    user = re.search(r"([\d.]+)\s+user", err)
    sys_ = re.search(r"([\d.]+)\s+sys", err)
    return {
        "wall_s": wall,
        "user_s": float(user.group(1)) if user else None,
        "sys_s": float(sys_.group(1)) if sys_ else None,
        "max_rss_bytes": grab("maximum resident set size"),
        "peak_footprint_bytes": grab("peak memory footprint"),
    }


def already_done() -> set:
    done = set()
    f = RESULTS / "results.jsonl"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["tool"], r["file"]))
    return done


def run_one(cfg: dict, fname: str, tname: str, repeats: int) -> dict:
    tool = tools(cfg)[tname]
    src = file_path(cfg, fname)
    if not src.exists():
        raise SystemExit(f"missing input {src}; run `bench.py prepare` first")
    nbytes, want = FILES[fname][1], FILES[fname][2]
    if want is None:
        want = sha256(src)
    if src.stat().st_size != nbytes:
        raise SystemExit(f"{src} has wrong size")
    scratch = WORK / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    packed = scratch / f"{fname}.{tname}.packed"
    restored = scratch / f"{fname}.{tname}.restored"
    if tname == "ppmd":
        packed = packed.with_suffix(".7z")
    enc_s, dec_s = [], []
    size = None
    for i in range(repeats):
        for p in (packed, restored):
            if p.exists():
                p.unlink()
        sub = lambda cmd: [a.replace("{in}", str(src)).replace("{out}", str(packed)).replace("{rest}", str(restored)) for a in cmd]
        print(f"  [{fname}/{tname}] run {i + 1}/{repeats}: encode ...", flush=True)
        e = timed(sub(tool["enc"]), packed if tool.get("stdout_enc") else None)
        if not packed.exists():  # e.g. paq8px may append its versioned extension
            cands = sorted(scratch.glob(packed.name + "*"))
            if len(cands) != 1:
                raise SystemExit(f"{tname}: cannot find compressed output {packed} (found {cands})")
            cands[0].rename(packed)
        size_i = packed.stat().st_size
        print(f"  [{fname}/{tname}] run {i + 1}/{repeats}: {size_i} B in {e['wall_s']:.1f} s; decode ...", flush=True)
        d = timed(sub(tool["dec"]), restored if tool.get("stdout_dec") else None)
        got = sha256(restored)
        if got != want:
            raise SystemExit(f"ROUND TRIP FAILED for {tname} on {fname}: {got} != {want}")
        if size is not None and size_i != size:
            print(f"  WARNING: compressed size changed between runs ({size} -> {size_i})", flush=True)
        size = size_i
        enc_s.append(e)
        dec_s.append(d)
        print(f"  [{fname}/{tname}] run {i + 1}/{repeats}: decode {d['wall_s']:.1f} s, exact", flush=True)
    for p in (packed, restored):
        if p.exists():
            p.unlink()

    def summary(samples):
        wall = statistics.median(s["wall_s"] for s in samples)
        rss = [s["max_rss_bytes"] for s in samples if s["max_rss_bytes"] is not None]
        foot = [s["peak_footprint_bytes"] for s in samples if s["peak_footprint_bytes"] is not None]
        return {
            "median_wall_s": wall,
            "kB_per_s": nbytes / wall / 1000,
            "max_rss_bytes": max(rss) if rss else None,
            "peak_footprint_bytes": max(foot) if foot else None,
            "samples": samples,
        }

    return {
        "tool": tname, "label": tool["label"], "threads": tool["threads"],
        "file": fname, "file_desc": FILES[fname][3], "input_bytes": nbytes,
        "compressed_bytes": size, "bpb": 8 * size / nbytes, "repeats": repeats,
        "round_trip": "exact (sha256)", "encode": summary(enc_s), "decode": summary(dec_s),
        "command_encode": " ".join(tool["enc"]), "command_decode": " ".join(tool["dec"]),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def cmd_run(stage: str, only: list[str]) -> None:
    cfg = load_config()
    RESULTS.mkdir(exist_ok=True)
    done = already_done()
    for fname, tname, reps in STAGES[stage]:
        if only and tname not in only:
            continue
        if (tname, fname) in done:
            print(f"skip {tname} on {fname} (already in results.jsonl)")
            continue
        print(f"== {tname} on {fname} x{reps}  [{time.strftime('%H:%M:%S')}]", flush=True)
        rec = run_one(cfg, fname, tname, reps)
        with (RESULTS / "results.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"   -> {rec['compressed_bytes']} B, {rec['bpb']:.4f} bpb, enc {rec['encode']['kB_per_s']:.2f} kB/s, "
              f"dec {rec['decode']['kB_per_s']:.2f} kB/s", flush=True)


SLICES = {"dev1": 909_000_000, "dev4": 909_000_000, "test1": 959_000_000,
          "enwik8": 0, "tail100": 900_000_000}


def cmd_prepare(enwik9: str | None) -> None:
    """Verify enwik9 and cut every benchmark file from it, checking each SHA-256."""
    cfg = load_config()
    if enwik9:
        cfg["enwik9"] = str(Path(enwik9).expanduser().resolve())
        CONFIG.write_text(json.dumps(cfg, indent=2) + "\n")
    e9 = Path(cfg["enwik9"])
    if not e9.exists():
        raise SystemExit(f"enwik9 not found at {e9}; get it from https://mattmahoney.net/dc/enwik9.zip")
    print("checking enwik9 sha256 ...", flush=True)
    assert sha256(e9) == FILES["enwik9"][2], "enwik9 checksum mismatch"
    WORK.mkdir(exist_ok=True)
    for name, off in SLICES.items():
        p = file_path(cfg, name)
        if not p.exists():
            with e9.open("rb") as f:
                f.seek(off)
                p.write_bytes(f.read(FILES[name][1]))
        assert sha256(p) == FILES[name][2], f"bad checksum {p}"
        print(f"ok {name}")
    (WORK / "smoke_20KB.bin").write_bytes(file_path(cfg, "dev1").read_bytes()[:20_000])
    print("ok smoke")


def sh(cmd: str) -> str:
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return (p.stdout.strip() or p.stderr.strip())
    except Exception as e:
        return f"<{e}>"


def cmd_env() -> None:
    cfg = load_config()
    RESULTS.mkdir(exist_ok=True)
    env = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "hw.model": sh("sysctl -n hw.model"),
        "cpu": sh("sysctl -n machdep.cpu.brand_string"),
        "perf_cores": sh("sysctl -n hw.perflevel0.physicalcpu"),
        "eff_cores": sh("sysctl -n hw.perflevel1.physicalcpu"),
        "l2_perf_cluster_bytes": sh("sysctl -n hw.perflevel0.l2cachesize"),
        "memsize_bytes": sh("sysctl -n hw.memsize"),
        "power": sh("pmset -g batt | head -1"),
        "clang": sh("clang++ --version | head -1"),
        "jolt_sha256": sha256(Path(cfg["jolt"])) if Path(cfg["jolt"]).exists() else "missing",
        "cmix_sha256": sha256(Path(cfg["cmix"])) if Path(cfg["cmix"]).exists() else "missing",
        "paq8px_sha256": sha256(Path(cfg["paq8px"])) if Path(cfg["paq8px"]).exists() else "missing",
        "xz": sh(f"{cfg['xz']} --version | head -1"),
        "zstd": sh(f"{cfg['zstd']} --version"),
        "bzip2": sh(f"{cfg['bzip2']} --help 2>&1 | head -1"),
        "7zz": sh(f"{cfg['sevenzip']} | head -2 | tail -1"),
        "config": cfg,
    }
    (RESULTS / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    print(json.dumps(env, indent=2))


def mib(x):
    return f"{x / 2**20:,.1f}" if x is not None else "?"


def cmd_table() -> None:
    f = RESULTS / "results.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["file"] != "smoke"]
    order = list(FILES)
    rows.sort(key=lambda r: (order.index(r["file"]), r["bpb"]))
    lines = ["| File | Compressor | Threads | Compressed bytes | bpb | Encode kB/s | Decode kB/s | Encode wall s | Decode wall s | Peak RSS MiB (enc/dec) | Runs |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        e, d = r["encode"], r["decode"]
        lines.append(f"| {r['file']} | {r['label']} | {r['threads']} | {r['compressed_bytes']:,} | {r['bpb']:.4f} | "
                     f"{e['kB_per_s']:,.2f} | {d['kB_per_s']:,.2f} | {e['median_wall_s']:,.1f} | {d['median_wall_s']:,.1f} | "
                     f"{mib(e['max_rss_bytes'])} / {mib(d['max_rss_bytes'])} | {r['repeats']} |")
    text = "\n".join(lines)
    (RESULTS / "table.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    verb = sys.argv[1]
    if verb == "env":
        cmd_env()
    elif verb == "prepare":
        arg = sys.argv[sys.argv.index("--enwik9") + 1] if "--enwik9" in sys.argv else None
        cmd_prepare(arg)
    elif verb == "run":
        stage = sys.argv[2]
        only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else []
        cmd_run(stage, only)
    elif verb == "table":
        cmd_table()
    else:
        print(__doc__)
        sys.exit(1)
