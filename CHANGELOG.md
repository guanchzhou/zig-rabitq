# Changelog

## 0.2.2 - 2026-10-25

- Fix a leak in `buildBits` when the second extended-code scratch allocation fails.
- `index.estimate` and `index.searchHits` now return errors: `error.IndexOutOfRange` for `i >= ix.n`, and `error.DimensionMismatch`, `error.CodeBitsOutOfRange` or `error.QueryBitsOutOfRange` when the index is inconsistent or the query was prepared for an index of another shape. Callers need `try`.
- `buildBits` returns `error.TooManyVectors` above `maxInt(u32)` vectors and `error.DimensionTooLarge` when the padded dimension overflows; buffer sizes use checked multiplication.
- `rotate.paddedLen` returns `error.DimensionTooLarge` instead of reaching `unreachable`, `rotate.apply` returns `error.DimensionMismatch` instead of asserting, and `rotate.applyInPlace` rotates an already padded buffer.
- Centring falls back to f64 where the f32 difference `x - c` overflows, so huge but finite inputs no longer produce infinite norms. Other results are unchanged.
- The one-bit exact-query estimate visits set code bits with `@ctz`; results are bit-identical.
- Allocation-failure tests over `buildBits`, `prepare` and `search`.

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
