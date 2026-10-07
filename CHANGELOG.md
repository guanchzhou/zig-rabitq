# Changelog

## 0.1.0 - 2026-10-07

- First release: one-bit RaBitQ codes with a seeded Hadamard rotation, the unbiased inner-product
  estimator and its error bound, 4-bit query quantisation for popcount scans, and top-k search.
- The `zig-rabitq` command encodes vectors and searches them, with optional exact re-ranking.
- Checked against an independent NumPy reference on 30 cases; fuzzed invariants.
