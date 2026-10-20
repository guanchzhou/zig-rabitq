# Changelog

## 0.2.1 - 2026-10-20

- Add zero-allocation `index.searchHits` returning `{ id, estimate }` via an in-place bounded max-heap of size `k`, and update `index.search` to allocate $O(k)$ instead of sorting all $O(n)$ items.
- Vectorise `rotate.fwht` with `@Vector(4, f64)`, bound CLI `stdin` reads, and add GitHub Actions CI workflow.

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
