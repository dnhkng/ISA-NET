# Published LTCB figures (not measured here)

Source: Large Text Compression Benchmark, https://mattmahoney.net/dc/text.html
Page header: "Last update: Nov. 19, 2026". Accessed 2026-11-22.
Times are LTCB ns/byte on the submitter's machine (see notes), NOT this M4. Memory in MB.

| LTCB rank | Program / options | enwik8 B | enwik9 B | Decompressor B | Total B | Comp ns/B | Decomp ns/B | Mem MB | Machine note |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | cmix-neif 2.0 | – | 89,346,392 | 0 | 89,346,392 | 177,048 | 177,480 | 9,770 | 119 |
| 22 | cmix v21 -t | 14,623,723 | 107,963,380 | 281,387 | 108,244,767 | 622,949 | 638,442 | 30,950 | 83 |
| 33 | paq8px_v206fix1 -12L | 15,849,084 | 124,696,410 | 402,949 | 125,099,359 | 291,916 | 294,847 | 28,151 | 93 |
| 48 | ppmonstr J -m1700 -o16 | 19,055,092 | 157,007,383 | 42,019 | 157,049,402 | 3,574 | ~3,600 | 1,700 | default (Athlon-64 3500+) |
| 89 | ppmd J1 -m256 -o10 -r1 | 21,388,296 | 183,964,915 | 11,099 | 183,976,014 | 880 | 895 | 256 | default (Athlon-64 3500+) |
| 105 | xz 5.2.1 --lzma2=preset=9e,dict=1GiB,lc=4,pb=0 | 24,703,772 | 197,331,816 | 36,752 | 197,368,568 | 5,876 | 20 | 6,000 | 73 |
| 125 | zstd 0.6.0 -22 --ultra | 25,405,601 | 215,674,670 | 69,687 | 215,744,357 | 701 | 2.2 | 792 | 76 |
| 152 | bzip2 1.0.2 -9 | 29,008,736 | 253,977,839 | 30,036 | 254,007,875 | 379 | 129 | 8 | default (Athlon-64 3500+) |

PPMd var. I: no row in the main table (only ppmd J1).

## Neighbourhood of Jolt's enwik9 total (161,855,306 B) — would sit between ranks 55 and 56

| Rank | Program | enwik9 B | Total B |
|---:|---|---:|---:|
| 48 | ppmonstr J -m1700 -o16 | 157,007,383 | 157,049,402 |
| 52 | bsc-m03 0.4.0 -b1000000000 | 160,258,936 | 160,364,392 |
| 53 | bwmonstr 0.02 | 160,468,597 | 160,537,998 |
| 54 | glza 0.12 c -x -o0.6 -p4 -r16000 | 161,678,356 | 161,678,356 (as printed; likely page typo) |
| 55 | nanozipltcb 0.09 | 161,581,290 | 161,715,074 |
| — | **ISA-NET Jolt (this work, measured on M4)** | 155,874,783 | 161,855,306 |
| 56 | kanzi -b 1024M -t RLT+TEXT+UTF -e TPAQX | 161,690,495 | 162,088,350 |
| 57 | M03 1.1b 1000000000 | 163,667,431 | 163,717,899 |

## Machine notes
- 83 (cmix v21): Intel Core i7-7700K, 32 GB DDR4, Ubuntu.
- 119 (cmix-neif 2.0): GCP c2d-highcpu-8, AMD EPYC 7B13, 16 GiB.
- 93 (paq8px -12L): Intel Core i9-12900KS (E-cores and HT disabled), 2x16 GB DDR5-6400, Windows 10.
- 73 (xz): 1.7 GHz Intel Core i7, 8 GB, Mac OS X 10.10.3.
- 76 (zstd): Intel Core i7-3930K @ 4.5 GHz, Linux.
- default: Compaq Presario 5440, Athlon-64 3500+ 2.188 GHz, 32-bit Windows XP.

Caveats: ranks 1–3 were not in the fetched text; values transcribed via an automated fetch, spot-check before publishing.
