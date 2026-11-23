# Same-machine compressor benchmark: ISA-NET Jolt vs cmix v21, paq8px, PPMd, xz, zstd, bzip2 (Apple M4 Pro)

Run date: 2026-11-22 (16:48–21:06 CET). Every row marked *measured* was produced in this session by `bench.py`, round-tripped
byte-exact (SHA-256 of the restored file = SHA-256 of the input), and is in `results/results.jsonl`. Rows marked
*published (LTCB)* or *extrapolated* were not measured here; their sources are given. Raw data: `results/` (results.jsonl,
table.md, environment.json, ltcb_published.md) and `logs/`.

**Training-data status, which matters for every Jolt number:** Jolt was pretrained on enwik9[0:900,000,000].
The dev1/dev4 slices (enwik9[909M:…]), test1 (enwik9[959M:960M]) and tail100 (enwik9[900M:1000M]) are **held out**.
enwik8 (= enwik9[0:100M]) and full enwik9 are **inside Jolt's training data** (corpus-tuned). All other compressors are
online/adaptive with no training on enwik.

---

## 1. Machine and setup

| Item | Value |
|---|---|
| Machine | Apple M4 Pro, 12 cores (8 performance + 4 efficiency), 24 GB RAM (25,769,803,776 B) |
| OS | macOS 27.2.1 (build 26B131) |
| Power | AC power for every timed run (checked every 30 s in `logs/monitor.log`; 0 samples on battery). `caffeinate -dimsu` throughout. |
| Compiler | Apple clang 21.0.0 (clang-2100.3.34.2) |
| cmix | v21 (kit source, fx2-cmix improvements). Built with `make CXX=clang++` (target `all`): `-std=c++17 -Wall -DP1_MODE=2 -DP3_BATCH -DP2_FASTMATH -DLSTM_QUANT=8 -Ofast -march=native -DNDEBUG`. `-march=native` was accepted; no fallback needed. SHA-256 `194cdf14…44e2`. |
| paq8px | v217 (kit source, commit c84f576). `cmake -DCMAKE_BUILD_TYPE=Release -DNATIVECPU=ON` → `-O3 -funroll-loops -ftree-vectorize -fgcse-sm(ignored by clang) -falign-loops=16 -fno-fast-math -ffp-contract=off -march=native -mtune=native`. SHA-256 `6ea6e483…d0e1`. **Two build fixes, see §7.** |
| paq8px level | **-11L** (not -12L): the Mac has 24 GB < 32 GB, as the kit README prescribes. Measured footprint at -11L: 10.9–11.3 GiB. |
| Jolt | `isa-net-m4`, SHA-256 `4a8c5899b4aafd74beec59db4bc855dd0f1d3e6a064f4f32984f92a52c136547` (identical to the enwik9 run's binary). Run from `~/Documents/isa-net/bin/isa-net-m4` (see §7). |
| PPMd | 7-Zip 25.01 (arm64) `7zz`, PPMd **var.H**, `o=16 mem=1g`, `-mmt=1` |
| xz / zstd / bzip2 | XZ Utils 5.8.3 (`-9e -T1`); zstd v1.5.7 (`--ultra -22 --long=30 -T1`); bzip2 (macOS system, `-9`) |
| Timing | wall clock, encode and decode separately, median of runs; memory = `maximum resident set size` and `peak memory footprint` from `/usr/bin/time -l` |

## 2. Main table: 1 MB dev slice (enwik9[909M:910M], held out from Jolt training) — measured

| Compressor | Threads | Compressed bytes | bpb | Encode kB/s | Decode kB/s | Peak RSS MiB (enc/dec) | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| cmix v21 + english.dic | 1 | 169,484 | 1.3559 | 1.59 | 1.78 | 7,893 / 9,930 ¹ | 1 |
| **ISA-NET Jolt, 1 worker** | 1 | 169,645 | 1.3572 | 60.73 | 60.62 | 13.9 / 14.5 | 5 |
| ISA-NET Jolt, 4 workers | 4 | 169,645 | 1.3572 | 219.83 | 219.03 | 14.0 / 13.4 | 5 |
| cmix v21, no dictionary | 1 | 187,913 | 1.5033 | 1.62 | 1.48 | 9,854 / 9,337 ¹ | 1 |
| paq8px v217 -11L | 1 | 188,139 | 1.5051 | 3.92 | 3.88 | 8,981 / 8,948 | 1 |
| PPMd var.H (7zz o=16, 1 GB) | 1 | 252,145 | 2.0172 | 15,000 | 13,628 | 20.6 / 19.7 | 5 |
| bzip2 -9 | 1 | 280,040 | 2.2403 | 20,200 | 39,845 | 7.4 / 4.5 | 5 |
| xz -9e | 1 | 284,480 | 2.2758 | 6,017 | 50,634 | 75.1 / 3.1 | 5 |
| zstd -22 --ultra --long=30 | 1 | 294,925 | 2.3594 | 6,712 | 73,783 | 22.2 / 3.6 | 5 |

¹ cmix's *peak memory footprint* was 18.4–18.6 GiB; macOS compressed the rest of its pages, see §6 and §7.

## 3. 4 MB dev slice and 1 MB test slice — measured

**4 MB dev slice** (enwik9[909M:913M], held out):

| Compressor | Threads | Compressed bytes | bpb | Encode kB/s | Decode kB/s | Peak RSS MiB (enc/dec) | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| cmix v21 + english.dic | 1 | 636,804 | 1.2736 | 1.70 | 1.84 | 7,305 / 9,923 ¹ | 1 |
| **ISA-NET Jolt, 1 worker** | 1 | 675,757 | 1.3515 | 60.91 | 59.14 | 18.5 / 21.2 | 3 |
| ISA-NET Jolt, 4 workers | 4 | 675,757 | 1.3515 | 214.09 | 215.94 | 20.7 / 20.1 | 3 |
| paq8px v217 -11L | 1 | 690,909 | 1.3818 | 4.08 | 4.20 | 11,120 / 11,566 | 1 |
| PPMd var.H | 1 | 928,740 | 1.8575 | 13,734 | 12,558 | 69.3 / 68.3 | 5 |
| xz -9e | 1 | 1,065,592 | 2.1312 | 4,765 | 85,015 | 101.0 / 5.9 | 5 |
| zstd -22 --ultra --long=30 | 1 | 1,097,555 | 2.1951 | 5,163 | 278,080 | 75.7 / 7.6 | 5 |
| bzip2 -9 | 1 | 1,103,613 | 2.2072 | 22,111 | 49,851 | 7.4 / 4.5 | 5 |

¹ footprint 18.8 GiB; about 0.55 GB of swap was used during this run (§7).

**1 MB test slice** (enwik9[959M:960M], Jolt's unseen test split):

| Compressor | Threads | Compressed bytes | bpb | Encode kB/s | Decode kB/s | Peak RSS MiB (enc/dec) | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| cmix v21 + english.dic | 1 | 179,816 | 1.4385 | 2.31 | 2.30 | 13,324 / 13,585 | 1 |
| **ISA-NET Jolt, 1 worker** | 1 | 180,600 | 1.4448 | 60.97 | 60.71 | 13.9 / 14.5 | 3 |
| paq8px v217 -11L | 1 | 199,073 | 1.5926 | 4.17 | 4.16 | 11,130 / 11,129 | 1 |
| PPMd var.H | 1 | 262,335 | 2.0987 | 14,893 | 13,623 | 20.6 / 19.7 | 5 |
| bzip2 -9 | 1 | 290,915 | 2.3273 | 20,280 | 34,277 | 7.4 / 4.5 | 5 |
| xz -9e | 1 | 296,676 | 2.3734 | 6,174 | 52,410 | 75.2 / 3.1 | 5 |
| zstd -22 --ultra --long=30 | 1 | 305,054 | 2.4404 | 6,940 | 83,723 | 22.2 / 3.6 | 5 |

(The kit's `slices` stage does not run Jolt 4-worker or cmix without dictionary on test1.)

**Sanity checks against earlier measurements:** Jolt 1-worker dev1 = 169,645 B = 1.357160 bpb, about 61 kB/s, about 14 MiB RSS
(release README: 1.357160 bpb, ~61 kB/s, ~14 MiB) ✔. Jolt test1 = 180,600 B, 1.4448 bpb (reference: 180,600 B, 1.4448) ✔.
cmix + dic dev1 1.3559 bpb (other machine: 1.3619); cmix no-dic 1.5033 (other machine: 1.5133) — small differences, plausibly
build/platform (fx2-cmix, ARM float paths). xz -9e dev1 2.2758 bpb (expected ~2.28) ✔. Jolt run-to-run wall-time spread ≤ 1.2 %
(1-worker) and ≤ 4 % (4-worker).

## 4. Large files

| File | Compressor | Threads | Compressed bytes | bpb | Encode kB/s | Decode kB/s | Peak RSS MiB (enc/dec) | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| enwik8 ⚠ in Jolt training | **Jolt, 4 workers** | 4 | 18,094,692 | 1.4476 | 220.6 | 220.6 | 258 / 237 | measured |
| enwik8 | PPMd var.H | 1 | 21,553,033 | 1.7242 | 6,253 | 5,848 | 1,029 / 1,029 | measured |
| enwik8 | xz -9e | 1 | 24,831,648 | 1.9865 | 2,003 | 126,274 | 674 / 66 | measured |
| enwik8 | zstd -22 --long=30 | 1 | 25,272,471 | 2.0218 | 2,041 | 1,036,461 | 830 / 100 | measured |
| enwik8 | bzip2 -9 | 1 | 29,008,758 | 2.3207 | 24,002 | 55,493 | 7.4 / 4.5 | measured |
| enwik8 | cmix v21 | 1 | 14,623,723 | 1.1699 | ≈1.70 (≈16.3 h) | ≈1.84 (≈15.1 h) | — (LTCB: 30,950 MB) | size **published (LTCB)**; time **extrapolated** |
| enwik8 | paq8px | 1 | 15,849,084 (v206fix1 -12L) | 1.2679 | ≈4.08 (≈6.8 h) | ≈4.20 (≈6.6 h) | — (LTCB: 28,151 MB) | size **published (LTCB)**; time **extrapolated (v217 -11L)** |
| enwik8 | PPMonstr var.J -m1700 -o16 | 1 | 19,055,092 | 1.5244 | — | — | — (LTCB: 1,700 MB) | **published (LTCB)** only; Windows binary, not run |
| tail100 (held out) | **Jolt, 4 workers** | 4 | 16,478,551 | 1.3183 | 220.5 | 220.4 | 261 / 234 | measured |
| tail100 | PPMd var.H | 1 | 19,583,734 | 1.5667 | 7,133 | 6,729 | 1,029 / 1,028 | measured |
| tail100 | xz -9e | 1 | 22,623,976 | 1.8099 | 2,144 | 135,425 | 674 / 66 | measured |
| tail100 | zstd -22 --long=30 | 1 | 23,095,219 | 1.8476 | 2,183 | 1,103,805 | 829 / 100 | measured |
| tail100 | bzip2 -9 | 1 | 26,738,048 | 2.1390 | 23,526 | 57,959 | 7.4 / 4.5 | measured |
| tail100 | cmix / paq8px / PPMonstr | 1 | — | — | — | — | — | not run (days of CPU); no published number exists for this file |
| enwik9 ⚠ 90 % in Jolt training | **Jolt, 4 workers** | 4 | 155,874,783 (161,855,306 with zipped decoder) | 1.2470 (1.2948) | 222.2 | 221.1 | 2,380 / 1,980 | measured **earlier on this Mac**, not in this session (`reference/isa_net_jolt_enwik9_full_parallel4.json`, same binary SHA) |
| enwik9 | PPMd var.H | 1 | 185,861,939 | 1.4869 | 7,326 | 6,913 | 1,029 / 1,029 | measured |
| enwik9 | zstd -22 --long=30 | 1 | 211,517,396 | 1.6921 | 1,780 | 1,237,873 | 2,242 / 958 | measured |
| enwik9 | xz -9e | 1 | 211,776,220 | 1.6942 | 1,978 | 153,090 | 675 / 66 | measured |
| enwik9 | bzip2 -9 | 1 | 253,977,891 | 2.0318 | 23,863 | 61,276 | 7.4 / 4.5 | measured |
| enwik9 | cmix v21 | 1 | 107,963,380 (total 108,244,767) | 0.8637 | ≈1.70 (≈163 h ≈ 6.8 days) | ≈1.84 (≈151 h ≈ 6.3 days) | — (LTCB: 30,950 MB) | size **published (LTCB)**; time **extrapolated** |
| enwik9 | paq8px | 1 | 124,696,410 (total 125,099,359; v206fix1 -12L) | 0.9976 | ≈4.08 (≈68 h ≈ 2.8 days) | ≈4.20 (≈66 h ≈ 2.8 days) | — (LTCB: 28,151 MB) | size **published (LTCB)**; time **extrapolated (v217 -11L)** |
| enwik9 | PPMonstr var.J -m1700 -o16 | 1 | 157,007,383 (total 157,049,402) | 1.2561 | — | — | — (LTCB: 1,700 MB) | **published (LTCB)** only |

*Extrapolation method:* `time ≈ bytes / (this Mac's measured kB/s on the 4 MB dev slice)`. This is **optimistic**: CM compressors
usually slow down as their models fill on large inputs, and the paq8px figure is for v217 -11L, not the -12L
version whose size is quoted. *Published:* LTCB, https://mattmahoney.net/dc/text.html, page "Last update: Nov. 19, 2026",
accessed 2026-11-22; LTCB times come from other machines (cmix v21: i7-7700K; paq8px: i9-12900KS; PPMonstr: Athlon-64 3500+), see
`results/ltcb_published.md`. Newer CM results exist on LTCB (cmix-neif 2.0, rank 5: 89,346,392 B on enwik9) but are not cmix v21.

**LTCB placement of Jolt's enwik9 total (161,855,306 B):** it falls between rank 55 (nanozipltcb 0.09, 161,715,074) and rank 56
(kanzi … TPAQX, 162,088,350), i.e. it would be **about 56th**, not "roughly 50th". PPMonstr var.J is rank 48 at 157,049,402 B,
4.81 MB smaller in total. On the payload alone, Jolt's 155,874,783 B is smaller than PPMonstr's 157,007,383 B; the 5.98 MB zipped
decoder (which includes the 7.05 MB model) is what puts Jolt behind. And enwik9 is 90 % inside Jolt's training data.

## 5. Speed ratios (measured, same Mac, single core: Jolt 1-worker vs CM compressors)

| File | vs | Jolt encode speed-up | Jolt decode speed-up | Jolt size vs other | bpb gap (Jolt − other) |
|---|---|---:|---:|---:|---:|
| dev1 (1 MB) | cmix + dic | **38.2×** | **34.0×** | +0.09 % larger | +0.0013 |
| dev1 | cmix, no dic | 37.4× | 40.8× | 9.72 % smaller | −0.1461 |
| dev1 | paq8px -11L | **15.5×** | **15.6×** | 9.83 % smaller | −0.1480 |
| dev4 (4 MB) | cmix + dic | **35.8×** | **32.2×** | +6.12 % larger | +0.0779 |
| dev4 | paq8px -11L | **14.9×** | **14.1×** | 2.19 % smaller | −0.0303 |
| test1 (1 MB) | cmix + dic | 26.4× | 26.4× | +0.44 % larger | +0.0063 |
| test1 | paq8px -11L | 14.6× | 14.6× | 9.28 % smaller | −0.1478 |

In words: **on this M4 Pro, single-core, Jolt is about 32–38× faster than cmix v21 (with dictionary) on the dev slices and its
output is 0.1 % larger on 1 MB and 6.1 % larger on 4 MB; it is about 14–16× faster than paq8px -11L and 2–10 % smaller.**
The gap to cmix widens with file length because cmix keeps adapting (dev4 bpb 1.2736 vs dev1 1.3559) while Jolt's pretrained
model is already near its level (1.3515 vs 1.3572). paq8px, run at -11L because of RAM, also improves with length.

cmix's speed here is held down by memory pressure (§6). On test1, where cmix kept 13 GiB resident (16 % system time instead of
28–37 %), it ran at 2.31 kB/s, and the Jolt ratio dropped to 26×. As a rough, **estimated (not measured)** bound: if all of cmix's
system time on dev4 were removed, its user time alone (1,462 s encode) would give about 2.7 kB/s, still ≈ 22× slower than Jolt.

## 6. Memory (peak, measured)

| Compressor | dev1 RSS MiB (enc/dec) | dev4 RSS MiB | test1 RSS MiB | Peak footprint |
|---|---:|---:|---:|---:|
| Jolt, 1 worker | 13.9 / 14.5 | 18.5 / 21.2 | 13.9 / 14.5 | ≈ RSS |
| Jolt, 4 workers | 14.0 / 13.4 | 20.7 / 20.1 | — | ≈ RSS; 258 MiB on 100 MB files; 2,380 MiB (2.32 GiB) on full enwik9 (earlier run) |
| cmix v21 + dic | 7,893 / 9,930 | 7,305 / 9,923 | 13,324 / 13,585 | **18.6–18.8 GiB** |
| cmix v21, no dic | 9,854 / 9,337 | — | — | 18.4 GiB |
| paq8px v217 -11L | 8,981 / 8,948 | 11,120 / 11,566 | 11,130 / 11,129 | 10.9–11.3 GiB |
| PPMd var.H (mem=1g) | 20.6 / 19.7 | 69.3 / 68.3 | 20.6 / 19.7 | 1.0 GiB on 100 MB+ files |
| xz -9e | 75.1 / 3.1 | 101 / 5.9 | 75.2 / 3.1 | 674 MiB on 100 MB+ files |
| zstd -22 --long=30 | 22.2 / 3.6 | 75.7 / 7.6 | 22.2 / 3.6 | 2.2 GiB on enwik9 |
| bzip2 -9 | 7.4 / 4.5 | 7.4 / 4.5 | 7.4 / 4.5 | 7.4 MiB |

For cmix, RSS understates its real demand: it allocates about 18.7 GiB up front, and on this 24 GB Mac macOS kept only 7–13 GiB
resident and compressed the rest. Jolt needs over 1,000× less memory than cmix on the slices (14 MiB vs 18.7 GiB footprint).

## 7. Caveats (plainly)

- **Jolt is pretrained on enwik9[0:900M].** Its enwik8 and enwik9 numbers are corpus-tuned (in-training). dev1, dev4, test1 and
  tail100 are held out. (Curiously, Jolt does worse on in-training enwik8, 1.4476 bpb, than on held-out tail100, 1.3183 bpb;
  this is measured, not explained here.)
- **Jolt's 7.05 MB model is side information inside the binary.** Slice and 100 MB sizes don't include it; the enwik9 total
  161,855,306 B does include the zipped decoder (5,980,523 B).
- **cmix "+ dic" rows use english.dic**, which is also side information, not counted in the slice sizes.
- **Jolt 4-worker rows use 4 cores** and the four-lane stream format; they are not single-core numbers. (On dev1/dev4 the 4-worker
  output is byte-for-byte the same size as 1-worker.) All ratios in §5 are single-core vs single-core.
- **paq8px was run at -11L, not -12L** (24 GB RAM < 32 GB, per the kit README). Published LTCB paq8px sizes are for v206fix1 -12L.
  -12L would likely compress somewhat better and run somewhat slower than what was measured here.
- **PPMd is var.H (7-Zip)**, not var.J/I as in LTCB; LTCB lists only `ppmd J1` in its main table (no var.I row).
- **xz on enwik9** here (211.8 MB, plain `-9e`) is larger than LTCB's xz row (197.3 MB), which used a 1 GiB dictionary with lc=4,pb=0.
- **Memory pressure and swap during cmix (the main anomaly).** cmix's 18.6–18.8 GiB footprint does not fit comfortably in 24 GB:
  23–37 % of its wall time on dev1/dev4 was system time (memory compression/paging), and about 0.55 GB of swap was used during
  cmix + dic on dev4 (swap started 18:08, peaked 554.75 MB at 18:34; it slowly drained back to ~206 MB and stayed there for the
  later runs, which belong to other processes being paged out). The kit README says not to let the machine swap during timed runs;
  it did here, once. Its effect on that run's speed looks small (dev4 cmix 1.70 kB/s vs dev1 1.59), but **cmix's speed on this Mac
  is memory-limited**: on test1 it ran 36–45 % faster (2.31 kB/s) with more of its memory resident. A Mac with ≥ 32 GB would likely
  give cmix a faster time, and Jolt's advantage over cmix would shrink toward the ~22–26× range. paq8px (≤ 11.3 GiB) was not
  affected (≤ 6 % system time).
- **cmix fixed start-up:** about 145 s each way on a 20 KB file with dictionary (12 s without), almost all of it start-up
  (`logs/smoke.log`). cmix speed is therefore judged only on the 1 MB and 4 MB slices; the start-up is included in those times.
- **Thermal throttling:** not observable — `pmset -g therm` reports no CPU speed limit on Apple Silicon. Jolt's repeated runs
  varied ≤ 1.2 % (1-worker), so no throttling signature appeared in the repeated measurements.
- **Single runs** for cmix and paq8px on every file (kit design); other tools 3–5 runs, median.
- **Changes to the kit:**
  - `paq8px/CMakeLists.txt`: `-floop-strip-mine` (GCC-only, rejected by Apple clang) is now added only when the compiler is GCC.
  - `paq8px/src/model/SimilarityModel.cpp:27` and `SimilarityModelPair.cpp:42`: `min(warmup, …)` → `std::min<size_t>(warmup, …)`.
    On macOS `size_t` (unsigned long) and `uint64_t` (unsigned long long) are distinct types, so paq8px's own `min()` overloads were
    ambiguous; on Linux the call resolves to the same 64-bit unsigned min, so behaviour is unchanged.
  - `bench_config.json`: `paq8px_level` = `-11L`; `jolt` = `~/Documents/isa-net/bin/isa-net-m4`. macOS Gatekeeper blocked the
    kit's quarantined (Safari-downloaded, unnotarized) copy of `jolt/isa-net-m4` and the file was removed from the kit. The
    `~/Documents` copy is the same binary (SHA-256 `4a8c5899…c136547`, verified) and was only executed, not modified. The first
    smoke attempt that failed on this is kept in `logs/smoke_attempt1_gatekeeper.log`; it wrote no results.
  - `bench.py` itself was not changed.
- **Not run:** optional `jolt_enwik9` (the earlier same-Mac, same-binary full run is cited instead) and optional `paq_enwik8`
  (extrapolated at ≈ 6.8 h encode + 6.6 h decode at -11L, so it would fit within the README's ~24 h limit; not started, pending
  David's go-ahead). cmix and paq8px were not run on tail100, enwik8 or enwik9 (≈ 15 h to ≈ 7 days each way).
- The kit's enwik9 was read from `~/Documents/TERNLOG/data/enwik9_corpus/enwik9` (read-only; SHA-256 verified).

## 8. Plain-English summary

On the same Apple M4 Pro, on held-out Wikipedia text, ISA-NET Jolt compresses about as well as cmix v21 — the strongest
compressor tested — on 1 MB pieces (within 0.1–0.4 %), but runs about 26–38 times faster single-core and uses about 14 MB of memory
instead of cmix's roughly 19 GB. On a 4 MB piece, cmix's continued learning gives it a 6 % edge. Jolt also beats paq8px (at -11L)
by 2–10 % in size while being about 15 times faster. PPMd, xz, zstd and bzip2 are roughly 100–400 times faster to compress
(and up to about 1,000 times faster to decompress), but their files are 37–74 % larger than Jolt's. The trade-off is that Jolt's knowledge comes from pretraining on the first 900 MB of
the same Wikipedia dump and lives in a 7 MB model: counting that model, its full-enwik9 result (161.9 MB) would rank about 56th on
the Large Text Compression Benchmark, behind PPMonstr's 157.0 MB.
