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

pub const Error = error{ DimensionMismatch, DimensionTooLarge };

/// The next power of two `>= dims`, at least 2; `error.DimensionTooLarge` if it does not fit in a `usize`.
pub fn paddedLen(dims: usize) Error!usize {
    return std.math.ceilPowerOfTwo(usize, @max(dims, 2)) catch error.DimensionTooLarge;
}

/// In-place unnormalised fast Walsh-Hadamard transform; `x.len` must be a power of two.
fn fwht(x: []f64) void {
    const Vec = @Vector(4, f64);
    var h: usize = 1;
    while (h < x.len) : (h *= 2) {
        var i: usize = 0;
        if (h >= 4) {
            while (i < x.len) : (i += 2 * h) {
                var j: usize = i;
                while (j < i + h) : (j += 4) {
                    const a: Vec = x[j..][0..4].*;
                    const b: Vec = x[j + h ..][0..4].*;
                    x[j..][0..4].* = a + b;
                    x[j + h ..][0..4].* = a - b;
                }
            }
        } else {
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
}

/// y = R x for `x` of any length up to `y.len`, which must be a power of two such as `paddedLen(x.len)`.
pub fn apply(seed: u64, x: []const f32, y: []f64) Error!void {
    if (y.len == 0 or !std.math.isPowerOfTwo(y.len) or x.len > y.len) return error.DimensionMismatch;
    for (y, 0..) |*v, i| v.* = if (i < x.len) x[i] else 0;
    transform(seed, y);
}

/// y = R y in place; `y.len` must be a power of two.
pub fn applyInPlace(seed: u64, y: []f64) Error!void {
    if (y.len == 0 or !std.math.isPowerOfTwo(y.len)) return error.DimensionMismatch;
    transform(seed, y);
}

fn transform(seed: u64, y: []f64) void {
    const n = y.len;
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(n)));
    for (0..rounds) |r| {
        for (y, 0..) |*v, i| {
            if (splitmix64(seed, r *% n +% i) & 1 == 1) v.* = -v.*;
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
    const n = try paddedLen(37);
    try std.testing.expectEqual(@as(usize, 64), n);
    var ya: [64]f64 = undefined;
    var yb: [64]f64 = undefined;
    try apply(7, &a, &ya);
    try apply(7, &b, &yb);
    var dx: f64 = 0;
    var dy: f64 = 0;
    for (a, b) |x, y| dx += @as(f64, x) * y;
    for (ya, yb) |x, y| dy += x * y;
    try std.testing.expectApproxEqAbs(dx, dy, 1e-9);
}

test "paddedLen and apply reject sizes they cannot handle" {
    try std.testing.expectEqual(@as(usize, 2), try paddedLen(0));
    try std.testing.expectEqual(@as(usize, 1) << 63, try paddedLen((@as(usize, 1) << 63) - 1));
    try std.testing.expectError(error.DimensionTooLarge, paddedLen((@as(usize, 1) << 63) + 1));
    try std.testing.expectError(error.DimensionTooLarge, paddedLen(std.math.maxInt(usize)));
    var y: [4]f64 = undefined;
    try std.testing.expectError(error.DimensionMismatch, apply(1, &.{ 1, 2, 3 }, y[0..3]));
    try std.testing.expectError(error.DimensionMismatch, apply(1, &.{ 1, 2, 3, 4, 5 }, &y));
    try std.testing.expectError(error.DimensionMismatch, apply(1, &.{}, y[0..0]));
    try std.testing.expectError(error.DimensionMismatch, applyInPlace(1, y[0..3]));
}

test "applyInPlace matches apply" {
    const x = [_]f32{ 1, -2, 3.5, 0.25, 7 };
    var a: [8]f64 = undefined;
    try apply(3, &x, &a);
    var b: [8]f64 = .{ 1, -2, 3.5, 0.25, 7, 0, 0, 0 };
    try applyInPlace(3, &b);
    try std.testing.expectEqualSlices(f64, &a, &b);
}
