//! RaBitQ with one bit per dimension (Gao and Long, SIGMOD 2024).
//!
//! Each data vector `o_r` is centred on `c`, normalised to `o = (o_r - c) / |o_r - c|` and rotated,
//! `y = R o`. Its code is the sign of each coordinate; the quantised unit vector it stands for is
//! `x = sign(y) / sqrt(D)`, with `D` the padded dimension. For a query `q` (centred, normalised,
//! rotated the same way) the estimator of `<o, q>` is `<x, R q> / <x, R o>`. It is unbiased, and with
//! high probability its error is below `sqrt((1 - <x,Ro>^2) / <x,Ro>^2) * eps0 / sqrt(D - 1)`.
//! Squared Euclidean distance follows from `|o_r - q_r|^2 = |o_r - c|^2 + |q_r - c|^2 - 2 |o_r - c| |q_r - c| <o, q>`.
//!
//! A query can be quantised to `bq` bits per coordinate with randomised rounding, so that `<x, Rq>`
//! becomes a sum of popcounts over bit planes; `bq = 0` keeps it exact.

const std = @import("std");
const rotate = @import("rotate.zig");

/// The paper's default confidence parameter.
pub const default_eps0 = 1.9;

pub const Error = error{ OutOfMemory, DimensionMismatch, EmptyIndex, NonFiniteValue, QueryBitsOutOfRange };

pub const Index = struct {
    n: usize,
    dims: usize,
    padded: usize,
    words: usize,
    seed: u64,
    centroid: []f32,
    codes: []u64,
    norms: []f64,
    dots: []f64,

    pub fn deinit(ix: Index, gpa: std.mem.Allocator) void {
        gpa.free(ix.centroid);
        gpa.free(ix.codes);
        gpa.free(ix.norms);
        gpa.free(ix.dots);
    }

    pub fn code(ix: Index, i: usize) []const u64 {
        return ix.codes[i * ix.words ..][0..ix.words];
    }
};

fn checkFinite(x: []const f32) Error!void {
    for (x) |v| if (!std.math.isFinite(v)) return error.NonFiniteValue;
}

/// Builds the index of `n = data.len / dims` row-major vectors. With `centroid == null` the mean
/// of the data is used.
pub fn build(gpa: std.mem.Allocator, data: []const f32, dims: usize, seed: u64, centroid: ?[]const f32) Error!Index {
    if (dims == 0 or data.len % dims != 0) return error.DimensionMismatch;
    const n = data.len / dims;
    if (n == 0) return error.EmptyIndex;
    try checkFinite(data);
    const padded = rotate.paddedLen(dims);
    const words = padded / 64 + @intFromBool(padded % 64 != 0);
    const c = try gpa.alloc(f32, dims);
    errdefer gpa.free(c);
    if (centroid) |cc| {
        if (cc.len != dims) return error.DimensionMismatch;
        try checkFinite(cc);
        @memcpy(c, cc);
    } else {
        var sum = try gpa.alloc(f64, dims);
        defer gpa.free(sum);
        @memset(sum, 0);
        for (0..n) |i| for (data[i * dims ..][0..dims], 0..) |v, j| {
            sum[j] += v;
        };
        for (c, sum) |*x, s| x.* = @floatCast(s / @as(f64, @floatFromInt(n)));
    }
    const codes = try gpa.alloc(u64, n * words);
    errdefer gpa.free(codes);
    @memset(codes, 0);
    const norms = try gpa.alloc(f64, n);
    errdefer gpa.free(norms);
    const dots = try gpa.alloc(f64, n);
    errdefer gpa.free(dots);
    const r = try gpa.alloc(f32, dims);
    defer gpa.free(r);
    const y = try gpa.alloc(f64, padded);
    defer gpa.free(y);
    const inv_sqrt_d = 1.0 / @sqrt(@as(f64, @floatFromInt(padded)));
    for (0..n) |i| {
        var nn: f64 = 0;
        for (r, data[i * dims ..][0..dims], c) |*v, x, m| {
            v.* = x - m;
            nn += @as(f64, v.*) * v.*;
        }
        norms[i] = @sqrt(nn);
        if (nn == 0) {
            dots[i] = 1;
            continue;
        }
        rotate.apply(seed, r, y);
        var abs_sum: f64 = 0;
        for (y, 0..) |v, j| {
            if (v > 0) codes[i * words + j / 64] |= @as(u64, 1) << @intCast(j % 64);
            abs_sum += @abs(v);
        }
        dots[i] = abs_sum / norms[i] * inv_sqrt_d;
    }
    return .{ .n = n, .dims = dims, .padded = padded, .words = words, .seed = seed, .centroid = c, .codes = codes, .norms = norms, .dots = dots };
}

pub const Query = struct {
    norm: f64,
    y: []f64,
    bq: u8,
    planes: []u64,
    low: f64,
    delta: f64,
    sum_u: f64,

    pub fn deinit(q: Query, gpa: std.mem.Allocator) void {
        gpa.free(q.y);
        gpa.free(q.planes);
    }
};

/// Centres, normalises and rotates `q`; with `bq > 0` also quantises it with randomised rounding
/// whose offsets come from `splitmix64(round_seed, i)`.
pub fn prepare(gpa: std.mem.Allocator, ix: Index, q: []const f32, bq: u8, round_seed: u64) Error!Query {
    if (q.len != ix.dims) return error.DimensionMismatch;
    if (bq > 8) return error.QueryBitsOutOfRange;
    try checkFinite(q);
    const r = try gpa.alloc(f32, ix.dims);
    defer gpa.free(r);
    var nn: f64 = 0;
    for (r, q, ix.centroid) |*v, x, m| {
        v.* = x - m;
        nn += @as(f64, v.*) * v.*;
    }
    const norm = @sqrt(nn);
    const y = try gpa.alloc(f64, ix.padded);
    errdefer gpa.free(y);
    rotate.apply(ix.seed, r, y);
    if (norm > 0) for (y) |*v| {
        v.* /= norm;
    };
    const planes = try gpa.alloc(u64, @as(usize, bq) * ix.words);
    errdefer gpa.free(planes);
    @memset(planes, 0);
    var low: f64 = 0;
    var delta: f64 = 0;
    var sum_u: f64 = 0;
    if (bq > 0) {
        low = std.mem.min(f64, y);
        const high = std.mem.max(f64, y);
        const levels: f64 = @floatFromInt((@as(u32, 1) << @intCast(bq)) - 1);
        delta = if (high > low) (high - low) / levels else 1;
        for (y, 0..) |v, i| {
            const offset = @as(f64, @floatFromInt(rotate.splitmix64(round_seed, i) >> 11)) * 0x1p-53;
            const u: u32 = @intFromFloat(@min(levels, @floor((v - low) / delta + offset)));
            sum_u += @floatFromInt(u);
            for (0..bq) |b| {
                if ((u >> @intCast(b)) & 1 == 1) planes[b * ix.words + i / 64] |= @as(u64, 1) << @intCast(i % 64);
            }
        }
    }
    return .{ .norm = norm, .y = y, .bq = bq, .planes = planes, .low = low, .delta = delta, .sum_u = sum_u };
}

pub const Estimate = struct {
    /// Estimate of <o, q> for the centred, normalised vectors.
    ip: f64,
    /// Half-width of the confidence interval for `ip` at the given eps0.
    bound: f64,
    /// Estimated squared Euclidean distance between the original vectors.
    dist2: f64,
};

pub fn estimate(ix: Index, q: Query, i: usize, eps0: f64) Estimate {
    const d: f64 = @floatFromInt(ix.padded);
    const norm = ix.norms[i];
    if (norm == 0 or q.norm == 0) return .{ .ip = 0, .bound = 0, .dist2 = norm * norm + q.norm * q.norm };
    const cw = ix.code(i);
    var on_sum: f64 = 0;
    var all_sum: f64 = 0;
    if (q.bq == 0) {
        for (q.y, 0..) |v, j| {
            if ((cw[j / 64] >> @intCast(j % 64)) & 1 == 1) on_sum += v;
            all_sum += v;
        }
    } else {
        var ones: u64 = 0;
        for (cw) |w| ones += @popCount(w);
        var weighted: f64 = 0;
        for (0..q.bq) |b| {
            var c: u64 = 0;
            for (cw, q.planes[b * ix.words ..][0..ix.words]) |w, p| c += @popCount(w & p);
            weighted += @as(f64, @floatFromInt(c)) * @as(f64, @floatFromInt(@as(u32, 1) << @intCast(b)));
        }
        on_sum = q.delta * weighted + q.low * @as(f64, @floatFromInt(ones));
        all_sum = q.delta * q.sum_u + q.low * d;
    }
    const xq = (2 * on_sum - all_sum) / @sqrt(d);
    const dot = ix.dots[i];
    const ip = xq / dot;
    const bound = @sqrt(@max(0, 1 - dot * dot) / (dot * dot)) * eps0 / @sqrt(d - 1);
    return .{ .ip = ip, .bound = bound, .dist2 = norm * norm + q.norm * q.norm - 2 * norm * q.norm * ip };
}

/// The `out.len` indices with the smallest estimated distance, nearest first; ties to the lower index.
pub fn search(gpa: std.mem.Allocator, ix: Index, q: Query, out: []u32) Error!usize {
    const k = @min(out.len, ix.n);
    const Item = struct { i: u32, d: f64 };
    const items = try gpa.alloc(Item, ix.n);
    defer gpa.free(items);
    for (items, 0..) |*it, i| it.* = .{ .i = @intCast(i), .d = estimate(ix, q, i, default_eps0).dist2 };
    std.mem.sort(Item, items, {}, struct {
        fn f(_: void, a: Item, b: Item) bool {
            return a.d < b.d or (a.d == b.d and a.i < b.i);
        }
    }.f);
    for (out[0..k], items[0..k]) |*o, it| o.* = it.i;
    return k;
}

fn randomData(gpa: std.mem.Allocator, n: usize, d: usize, seed: u64) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const x = try gpa.alloc(f32, n * d);
    for (x) |*v| v.* = r.floatNorm(f32);
    return x;
}

test "codes and inner products" {
    const gpa = std.testing.allocator;
    const d = 96;
    const data = try randomData(gpa, 50, d, 3);
    defer gpa.free(data);
    const ix = try build(gpa, data, d, 11, null);
    defer ix.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 128), ix.padded);
    try std.testing.expectEqual(@as(usize, 2), ix.words);
    for (ix.dots) |v| try std.testing.expect(v > 0.6 and v <= 1); // about sqrt(2/π) for random directions
    const q = try prepare(gpa, ix, data[0..d], 0, 0);
    defer q.deinit(gpa);
    // The query equals data vector 0, so its own estimate is exactly 1.
    try std.testing.expectApproxEqAbs(@as(f64, 1), estimate(ix, q, 0, default_eps0).ip, 1e-12);
    const q4 = try prepare(gpa, ix, data[d .. 2 * d], 4, 99);
    defer q4.deinit(gpa);
    const qx = try prepare(gpa, ix, data[d .. 2 * d], 0, 0);
    defer qx.deinit(gpa);
    for (0..ix.n) |i| {
        const a = estimate(ix, q4, i, default_eps0).ip;
        const b = estimate(ix, qx, i, default_eps0).ip;
        try std.testing.expect(@abs(a - b) < 0.1);
    }
}

test "the estimator is unbiased and its spread matches the bound" {
    // The error is close to normal with standard deviation bound / eps0, so eps0 = 1.9 covers about
    // 94 percent and eps0 = 3 about 99.7 percent.
    const gpa = std.testing.allocator;
    const d = 256;
    const data = try randomData(gpa, 400, d, 5);
    defer gpa.free(data);
    const ix = try build(gpa, data, d, 13, null);
    defer ix.deinit(gpa);
    var covered19: usize = 0;
    var covered3: usize = 0;
    var total: usize = 0;
    var bias: f64 = 0;
    var sq: f64 = 0;
    var sigma: f64 = 0;
    for (0..4) |qi| {
        const qv = data[(399 - qi) * d ..][0..d];
        const q = try prepare(gpa, ix, qv, 0, 0);
        defer q.deinit(gpa);
        var qn: f64 = 0;
        for (qv, ix.centroid) |x, m| qn += (@as(f64, x) - m) * (@as(f64, x) - m);
        for (0..ix.n - 4) |i| {
            var dot: f64 = 0;
            for (data[i * d ..][0..d], qv, ix.centroid) |a, b, m| dot += (@as(f64, a) - m) * (@as(f64, b) - m);
            const truth = dot / (ix.norms[i] * @sqrt(qn));
            const e = estimate(ix, q, i, 1);
            const err = e.ip - truth;
            covered19 += @intFromBool(@abs(err) <= 1.9 * e.bound);
            covered3 += @intFromBool(@abs(err) <= 3 * e.bound);
            bias += err;
            sq += err * err;
            sigma += e.bound * e.bound;
            total += 1;
        }
    }
    const t: f64 = @floatFromInt(total);
    try std.testing.expect(@abs(bias / t) < 0.005);
    try std.testing.expect(@abs(@sqrt(sq / t) / @sqrt(sigma / t) - 1) < 0.1);
    try std.testing.expect(@as(f64, @floatFromInt(covered19)) / t > 0.92);
    try std.testing.expect(@as(f64, @floatFromInt(covered3)) / t > 0.99);
}

test "search ranks the query's own vector first" {
    const gpa = std.testing.allocator;
    const d = 64;
    const data = try randomData(gpa, 30, d, 8);
    defer gpa.free(data);
    const ix = try build(gpa, data, d, 1, null);
    defer ix.deinit(gpa);
    const q = try prepare(gpa, ix, data[7 * d ..][0..d], 4, 2);
    defer q.deinit(gpa);
    var out: [5]u32 = undefined;
    try std.testing.expectEqual(@as(usize, 5), try search(gpa, ix, q, &out));
    try std.testing.expectEqual(@as(u32, 7), out[0]);
}

test "rejects bad input" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.DimensionMismatch, build(gpa, &.{ 1, 2, 3 }, 2, 0, null));
    try std.testing.expectError(error.EmptyIndex, build(gpa, &.{}, 2, 0, null));
    try std.testing.expectError(error.NonFiniteValue, build(gpa, &.{ 1, std.math.inf(f32) }, 2, 0, null));
}
