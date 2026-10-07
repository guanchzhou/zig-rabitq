//! A seeded random orthogonal transform: three rounds of random signs followed by a fast
//! Walsh-Hadamard transform, `R = H D3 H D2 H D1` scaled to be orthonormal. Inputs are padded with
//! zeros to the next power of two. The signs of round `r` come from `splitmix64(seed, r * n + i)`,
//! bit 0, so the transform is reproducible on every platform.

const std = @import("std");

pub fn splitmix64(seed: u64, i: u64) u64 {
    var z = seed +% (i +% 1) *% 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

pub const rounds = 3;

pub fn paddedLen(dims: usize) usize {
    return std.math.ceilPowerOfTwo(usize, @max(dims, 2)) catch unreachable;
}

/// In-place unnormalised fast Walsh-Hadamard transform; `x.len` must be a power of two.
fn fwht(x: []f64) void {
    var h: usize = 1;
    while (h < x.len) : (h *= 2) {
        var i: usize = 0;
        while (i < x.len) : (i += 2 * h) {
            for (i..i + h) |j| {
                const a = x[j];
                const b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

/// y = R x for `x` of any length up to `y.len`, which must be `paddedLen(x.len)`.
pub fn apply(seed: u64, x: []const f32, y: []f64) void {
    const n = y.len;
    std.debug.assert(std.math.isPowerOfTwo(n) and x.len <= n);
    for (y, 0..) |*v, i| v.* = if (i < x.len) x[i] else 0;
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(n)));
    for (0..rounds) |r| {
        for (y, 0..) |*v, i| {
            if (splitmix64(seed, r * n + i) & 1 == 1) v.* = -v.*;
        }
        fwht(y);
        for (y) |*v| v.* *= scale;
    }
}

test "orthogonal: preserves norms and inner products" {
    var prng = std.Random.DefaultPrng.init(1);
    const r = prng.random();
    var a: [37]f32 = undefined;
    var b: [37]f32 = undefined;
    for (&a, &b) |*x, *y| {
        x.* = r.floatNorm(f32);
        y.* = r.floatNorm(f32);
    }
    const n = paddedLen(37);
    try std.testing.expectEqual(@as(usize, 64), n);
    var ya: [64]f64 = undefined;
    var yb: [64]f64 = undefined;
    apply(7, &a, &ya);
    apply(7, &b, &yb);
    var dx: f64 = 0;
    var dy: f64 = 0;
    for (a, b) |x, y| dx += @as(f64, x) * y;
    for (ya, yb) |x, y| dy += x * y;
    try std.testing.expectApproxEqAbs(dx, dy, 1e-9);
}
