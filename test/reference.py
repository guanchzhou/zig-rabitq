"""Independent reference outputs for zig-rabitq, written to test/golden.json.

NumPy only; no code is shared with the Zig implementation. The rotation is materialised as a dense
matrix (the transform applied to each basis vector) and checked to be orthogonal before use, so
codes and estimates come from matrix products rather than an in-place Hadamard transform.

Run: uv run --with numpy python test/reference.py
"""

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "golden.json"
MASK = (1 << 64) - 1
RNG = np.random.default_rng(20261007)


def splitmix64(seed: int, i: int) -> int:
    z = (seed + ((i + 1) * 0x9E3779B97F4A7C15)) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def rotation(seed: int, n: int) -> np.ndarray:
    H = hadamard(n) / np.sqrt(n)
    R = np.eye(n)
    for r in range(3):
        D = np.diag([-1.0 if splitmix64(seed, r * n + i) & 1 else 1.0 for i in range(n)])
        R = H @ D @ R
    assert np.allclose(R.T @ R, np.eye(n), atol=1e-12)
    return R


def padded(d: int) -> int:
    n = 2
    while n < d:
        n *= 2
    return n


def case() -> dict:
    n = int(RNG.integers(5, 61))
    d = int(RNG.integers(3, 131))
    seed = int(RNG.integers(0, 2**63))
    X = RNG.normal(size=(n, d)).astype(np.float32)
    if RNG.random() < 0.3:
        X[int(RNG.integers(0, n))] = X.mean(axis=0) * 0  # a row that may sit near the centroid
    c = X.astype(np.float64).mean(axis=0).astype(np.float32)
    D = padded(d)
    R = rotation(seed, D)
    words = (D + 63) // 64
    codes, norms, dots = [], [], []
    for row in X:
        r = row.astype(np.float32) - c
        nn = float(np.dot(r.astype(np.float64), r.astype(np.float64)))
        norms.append(np.sqrt(nn))
        w = [0] * words
        if nn == 0:
            dots.append(1.0)
            codes.append(w)
            continue
        y = R @ np.pad(r.astype(np.float64), (0, D - d))
        for j, v in enumerate(y):
            if v > 0:
                w[j // 64] |= 1 << (j % 64)
        codes.append(w)
        dots.append(float(np.abs(y).sum() / norms[-1] / np.sqrt(D)))
    queries = []
    for _ in range(3):
        qv = RNG.normal(size=d).astype(np.float32)
        r = qv - c
        qn = float(np.sqrt(np.dot(r.astype(np.float64), r.astype(np.float64))))
        y = R @ np.pad(r.astype(np.float64), (0, D - d)) / qn
        out = {"q": qv.tolist()}
        for bq in (0, 4):
            seed_q = int(RNG.integers(0, 2**63))
            if bq:
                lo, hi = y.min(), y.max()
                levels = 2**bq - 1
                delta = (hi - lo) / levels if hi > lo else 1.0
                off = np.array([(splitmix64(seed_q, i) >> 11) * 2.0**-53 for i in range(D)])
                u = np.minimum(levels, np.floor((y - lo) / delta + off))
                yq = lo + delta * u
            else:
                yq = y
            ests = []
            for i in range(n):
                if norms[i] == 0:
                    ests.append({"ip": 0.0, "bound": 0.0, "dist2": norms[i] ** 2 + qn**2})
                    continue
                bits = np.array([(codes[i][j // 64] >> (j % 64)) & 1 for j in range(D)], dtype=float)
                xq = float(((2 * bits - 1) * yq).sum() / np.sqrt(D))
                ip = xq / dots[i]
                bound = np.sqrt(max(0.0, 1 - dots[i] ** 2) / dots[i] ** 2) * 1.9 / np.sqrt(D - 1)
                ests.append({"ip": ip, "bound": float(bound), "dist2": norms[i] ** 2 + qn**2 - 2 * norms[i] * qn * ip})
            order = sorted(range(n), key=lambda i: (ests[i]["dist2"], i))[:5]
            out[f"bq{bq}"] = {"round_seed": str(seed_q), "estimates": ests, "top5": order}
        queries.append(out)
    return {"d": d, "seed": str(seed), "data": X.ravel().tolist(), "padded": D,
            "codes": [[format(w, "016x") for w in ws] for ws in codes], "norms": norms, "dots": dots,
            "queries": queries}


def main():
    cases = [case() for _ in range(30)]
    OUT.write_text(json.dumps({"cases": cases}, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
