# zig-rabitq

RaBitQ binary codes for approximate nearest-neighbour search, in pure Zig 0.17. Each vector is
stored as one bit per dimension plus two numbers, and inner products and distances are estimated
from those bits with a known error bound.

- **Codes**: centre on the data mean, rotate with a seeded random orthogonal transform (random
  signs and Hadamard, three rounds), keep the sign of each coordinate.
- **Estimator**: `<x, Rq> / <x, Ro>`, unbiased for `<o, q>`, with error below
  `sqrt((1 - <x,Ro>^2) / <x,Ro>^2) * eps0 / sqrt(D - 1)` with high probability (Gao and Long, SIGMOD
  2024). Measured here, the error is close to normal with that standard deviation at `eps0 = 1`.
- **Scan**: queries can be quantised to 4 bits with randomised rounding, so each estimate is a few
  popcounts over bit planes.
- No dependencies and no C. Checked against an independent NumPy reference that builds the rotation
  as a dense matrix. MIT licensed.

## Install

```sh
zig fetch --save <url or path of zig-rabitq>
```

```zig
const rabitq = b.dependency("zig_rabitq", .{}).module("rabitq");
exe.root_module.addImport("rabitq", rabitq);
```

## Library

```zig
const ix = try rabitq.index.build(gpa, data, dims, seed, null); // null: centre on the data mean
defer ix.deinit(gpa);
const q = try rabitq.index.prepare(gpa, ix, query, 4, round_seed); // 4-bit query, 0 for exact
defer q.deinit(gpa);
const e = rabitq.index.estimate(ix, q, i, rabitq.index.default_eps0); // .ip, .bound, .dist2
_ = try rabitq.index.search(gpa, ix, q, top_k); // nearest first by estimated distance
```

Inputs are padded with zeros to the next power of two. A vector equal to the centroid has no
direction; its inner-product estimate is 0 and its distance is exact. Re-rank the best estimates
by exact distance when exact order matters; the command line does this with `rerank`.

[`examples/search.zig`](examples/search.zig) indexes 500 random 128-dimensional vectors and finds a
perturbed copy of one of them first with a 4-bit query scan.

## Command line

`zig build` installs `zig-out/bin/zig-rabitq`. Each command reads one JSON object on stdin and
writes one JSON object on stdout.

```sh
$ echo '{"data":[[1,0,0],[0.9,0.1,0],[0,1,0],[0,0,1],[-1,0,0]],"queries":[[1,0.05,0]],"k":3,"rerank":5}' | zig-rabitq search
{"results":[[0,1,2]]}
```

`zig-rabitq --help` lists the fields. Invalid input exits with 1; an unknown or missing command
exits with 2.

## Development

```sh
zig build test              # unit, golden, fuzz-corpus, example, and command-line tests
zig build test --fuzz       # fuzz the invariants (needs .zig-cache/tmp to exist)
zig build docs              # API documentation in zig-out/docs
uv run --with numpy python test/reference.py   # regenerate test/golden.json
```

`test/reference.py` materialises the rotation as a dense matrix, checks that it is orthogonal, and
computes codes, norms, `<x, Ro>`, estimates in exact and 4-bit query mode, and top-5 orders by
matrix products. `test/golden.zig` requires identical codes and orders and values within 1e-9 on
30 cases; a relative change of 5e-9 in the estimator makes it fail.

## Credits

- J. Gao and C. Long, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
  for Approximate Nearest Neighbor Search", *SIGMOD* (2024).
- N. Ailon and B. Chazelle, "The Fast Johnson-Lindenstrauss Transform and Approximate Nearest
  Neighbors", *SIAM J. Comput.* 39(1), 302-322 (2009), for randomised Hadamard rotations.

## License

MIT. See [LICENSE](LICENSE).
