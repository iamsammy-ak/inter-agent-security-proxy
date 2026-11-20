# Latency per message

Windows AMD64, Python 3.13.2; single process, CPU only; best of 3 runs per message, then percentiles across messages.

## short messages: 376 messages (median 100 characters, max 632)

| Stage | p50 (ms) | p95 (ms) | mean (ms) |
|---|---:|---:|---:|
| pii masking | 0.35 | 0.84 | 0.39 |
| rules + hidden-text + decoding | 0.63 | 1.97 | 0.83 |
| classifier (bge-small) | 32.43 | 88.13 | 38.73 |
| classifier (hashing) | 2.09 | 6.35 | 2.72 |
| full screen with bge-small (detect + mask + spotlight) | 34.40 | 111.42 | 44.39 |
| full screen with hashing (detect + mask + spotlight) | 3.69 | 10.54 | 4.47 |

## long tool results: 24 messages (median 5030 characters, max 6063)

| Stage | p50 (ms) | p95 (ms) | mean (ms) |
|---|---:|---:|---:|
| pii masking | 19.66 | 23.46 | 19.40 |
| rules + hidden-text + decoding | 37.63 | 57.87 | 37.26 |
| classifier (bge-small) | 3315.63 | 4808.84 | 3284.07 |
| classifier (hashing) | 370.50 | 699.31 | 401.10 |
| full screen with bge-small (detect + mask + spotlight) | 5624.25 | 7800.81 | 5216.26 |
| full screen with hashing (detect + mask + spotlight) | 184.24 | 270.47 | 193.50 |

Reproduce: `python benchmarks/run_benchmarks.py --suite latency`
