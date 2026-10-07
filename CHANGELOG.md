# Changelog

## 0.2.0 - 2026-10-07

- Extended RaBitQ: `buildBits` stores 1 to 8 bits per dimension, choosing for each vector the grid
  point with the best cosine by an exact breakpoint sweep. Estimates use popcounts over code and
  query bit planes. One-bit results are unchanged.
- The command line takes `bits` for `encode` and `search`.
- Golden cases now cover 2, 3, 4 and 8 bits; the fuzz target draws the bit count.

## 0.1.0 - 2026-10-07

- First release: one-bit RaBitQ codes with a seeded Hadamard rotation, the unbiased inner-product
  estimator and its error bound, 4-bit query quantisation for popcount scans, and top-k search.
- The `zig-rabitq` command encodes vectors and searches them, with optional exact re-ranking.
- Checked against an independent NumPy reference on 30 cases; fuzzed invariants.
