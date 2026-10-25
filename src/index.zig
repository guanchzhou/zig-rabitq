//! RaBitQ (Gao and Long, SIGMOD 2024) and extended RaBitQ with several bits per dimension (Gao et
//! al., SIGMOD 2025).
//!
//! Each data vector `o_r` is centred on `c`, normalised to `o = (o_r - c) / |o_r - c|` and rotated,
//! `y = R o`. Its code is the sign of each coordinate; the quantised unit vector it stands for is
//! `x = sign(y) / sqrt(D)`, with `D` the padded dimension. For a query `q` (centred, normalised,
//! rotated the same way) the estimator of `<o, q>` is `<x, R q> / <x, R o>`. It is unbiased, and with
//! high probability its error is below `sqrt((1 - <x,Ro>^2) / <x,Ro>^2) * eps0 / sqrt(D - 1)`.
//! Squared Euclidean distance follows from `|o_r - q_r|^2 = |o_r - c|^2 + |q_r - c|^2 - 2 |o_r - c| |q_r - c| <o, q>`.
//!
//! With `B > 1` bits per dimension the code is a point of the symmetric grid `sign(y) (m + 1/2)`,
//! `m` in `0 .. 2^(B-1) - 1`, scaled to the unit sphere: `m_i = min(floor(t |y_i|), 2^(B-1) - 1)` for
//! the scale `t` that maximises the cosine with `y`. The scale is found exactly by sweeping the
//! breakpoints `t = k / |y_i|` in increasing order (ties by coordinate) and updating the cosine
//! incrementally. `B = 1` is the sign code above. The estimator and its bound keep the same form.
//!
//! A query can be quantised to `bq` bits per coordinate with randomised rounding, so that `<x, Rq>`
//! becomes a sum of popcounts over bit planes; `bq = 0` keeps it exact.

const std = @import("std");
const rotate = @import("rotate.zig");

/// The paper's default confidence parameter.
pub const default_eps0 = 1.9;

pub const Error = error{ OutOfMemory, DimensionMismatch, DimensionTooLarge, EmptyIndex, NonFiniteValue, QueryBitsOutOfRange, CodeBitsOutOfRange, TooManyVectors, IndexOutOfRange };

pub const Index = struct {
    n: usize,
    dims: usize,
    padded: usize,
    words: usize,
    seed: u64,
    /// Bits per dimension of the data codes.
    bits: u8 = 1,
    centroid: []f32,
    /// Bit planes per vector: `bits` planes of `words` words, plane `b` holding bit `b` of each level.
    codes: []u64,
    norms: []f64,
    dots: []f64,
    /// For `bits > 1`: the norm of the grid point and the sum of its levels.
    gnorms: []f64 = &.{},
    sums: []f64 = &.{},

    pub fn deinit(ix: Index, gpa: std.mem.Allocator) void {
        gpa.free(ix.centroid);
        gpa.free(ix.codes);
        gpa.free(ix.norms);
        gpa.free(ix.dots);
        gpa.free(ix.gnorms);
        gpa.free(ix.sums);
    }

    pub fn code(ix: Index, i: usize) []const u64 {
        const w = @as(usize, ix.bits) * ix.words;
        return ix.codes[i * w ..][0..w];
    }
};

fn checkFinite(x: []const f32) Error!void {
    for (x) |v| if (!std.math.isFinite(v)) return error.NonFiniteValue;
}

/// Builds the one-bit index of `n = data.len / dims` row-major vectors. With `centroid == null` the
/// mean of the data is used.
pub fn build(gpa: std.mem.Allocator, data: []const f32, dims: usize, seed: u64, centroid: ?[]const f32) Error!Index {
    return buildBits(gpa, data, dims, seed, centroid, 1);
}

/// As `build`, with `bits` (1 to 8) bits per dimension. At most `maxInt(u32)` vectors, since search
/// reports `u32` ids.
pub fn buildBits(gpa: std.mem.Allocator, data: []const f32, dims: usize, seed: u64, centroid: ?[]const f32, bits: u8) Error!Index {
    if (bits == 0 or bits > 8) return error.CodeBitsOutOfRange;
    if (dims == 0 or data.len % dims != 0) return error.DimensionMismatch;
    const padded = try rotate.paddedLen(dims);
    const n = data.len / dims;
    if (n == 0) return error.EmptyIndex;
    if (n > std.math.maxInt(u32)) return error.TooManyVectors;
    try checkFinite(data);
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
    const codes = try gpa.alloc(u64, try mulSize(&.{ n, bits, words }));
    errdefer gpa.free(codes);
    @memset(codes, 0);
    const gnorms = try gpa.alloc(f64, if (bits > 1) n else 0);
    errdefer gpa.free(gnorms);
    const sums = try gpa.alloc(f64, if (bits > 1) n else 0);
    errdefer gpa.free(sums);
    var ex: ExScratch = if (bits > 1) try .init(gpa, padded, bits) else undefined;
    defer if (bits > 1) ex.deinit(gpa);
    const norms = try gpa.alloc(f64, n);
    errdefer gpa.free(norms);
    const dots = try gpa.alloc(f64, n);
    errdefer gpa.free(dots);
    const y = try gpa.alloc(f64, padded);
    defer gpa.free(y);
    const inv_sqrt_d = 1.0 / @sqrt(@as(f64, @floatFromInt(padded)));
    for (0..n) |i| {
        const nn = centre(y, data[i * dims ..][0..dims], c);
        norms[i] = @sqrt(nn);
        if (nn == 0) {
            dots[i] = 1;
            if (bits > 1) {
                gnorms[i] = 1;
                sums[i] = 0;
            }
            continue;
        }
        try rotate.applyInPlace(seed, y);
        if (bits > 1) {
            for (y) |*v| v.* /= norms[i];
            const res = ex.encode(y, codes[i * bits * words ..][0 .. bits * words], words);
            dots[i] = res.dot;
            gnorms[i] = res.gnorm;
            sums[i] = res.sum;
            continue;
        }
        var abs_sum: f64 = 0;
        for (y, 0..) |v, j| {
            if (v > 0) codes[i * words + j / 64] |= @as(u64, 1) << @intCast(j % 64);
            abs_sum += @abs(v);
        }
        dots[i] = abs_sum / norms[i] * inv_sqrt_d;
    }
    return .{ .n = n, .dims = dims, .padded = padded, .words = words, .seed = seed, .bits = bits, .centroid = c, .codes = codes, .norms = norms, .dots = dots, .gnorms = gnorms, .sums = sums };
}

fn mulSize(factors: []const usize) Error!usize {
    var p: usize = 1;
    for (factors) |f| p = std.math.mul(usize, p, f) catch return error.OutOfMemory;
    return p;
}

/// Writes `x - m` into `y[0..x.len]` and zeros after it; returns its squared norm. The difference
/// is taken in f32, as test/reference.py does, and in f64 only where the f32 one overflows.
fn centre(y: []f64, x: []const f32, m: []const f32) f64 {
    var nn: f64 = 0;
    for (y[0..x.len], x, m) |*v, a, b| {
        const narrow = a - b;
        v.* = if (std.math.isFinite(narrow)) narrow else @as(f64, a) - b;
        nn += v.* * v.*;
    }
    @memset(y[x.len..], 0);
    return nn;
}

const Breakpoint = struct { t: f64, i: u32 };

/// Exact scale search for extended RaBitQ codes.
const ExScratch = struct {
    half: u32,
    points: []Breakpoint,
    level: []u32,

    fn init(gpa: std.mem.Allocator, padded: usize, bits: u8) Error!ExScratch {
        const half = @as(u32, 1) << @intCast(bits - 1);
        const points = try gpa.alloc(Breakpoint, try mulSize(&.{ padded, half - 1 }));
        errdefer gpa.free(points);
        return .{ .half = half, .points = points, .level = try gpa.alloc(u32, padded) };
    }

    fn deinit(s: ExScratch, gpa: std.mem.Allocator) void {
        gpa.free(s.points);
        gpa.free(s.level);
    }

    fn before(_: void, a: Breakpoint, b: Breakpoint) bool {
        return a.t < b.t or (a.t == b.t and a.i < b.i);
    }

    /// Writes the bit planes of the best grid point for unit vector `y`.
    fn encode(s: ExScratch, y: []const f64, planes: []u64, words: usize) struct { dot: f64, gnorm: f64, sum: f64 } {
        var np: usize = 0;
        var num: f64 = 0;
        for (y, 0..) |v, i| {
            const a = @abs(v);
            num += 0.5 * a;
            if (a == 0) continue;
            for (1..s.half) |k| {
                s.points[np] = .{ .t = @as(f64, @floatFromInt(k)) / a, .i = @intCast(i) };
                np += 1;
            }
        }
        std.mem.sort(Breakpoint, s.points[0..np], {}, before);
        var den2: f64 = 0.25 * @as(f64, @floatFromInt(y.len));
        var best = num / @sqrt(den2);
        var best_at: usize = 0;
        @memset(s.level, 0);
        for (s.points[0..np], 1..) |p, step| {
            s.level[p.i] += 1;
            num += @abs(y[p.i]);
            den2 += 2 * @as(f64, @floatFromInt(s.level[p.i]));
            const c = num / @sqrt(den2);
            if (c > best) {
                best = c;
                best_at = step;
            }
        }
        @memset(s.level, 0);
        for (s.points[0..best_at]) |p| s.level[p.i] += 1;
        var g2: f64 = 0;
        var gy: f64 = 0;
        var sum: f64 = 0;
        @memset(planes, 0);
        for (y, s.level, 0..) |v, m, i| {
            const u: u32 = if (v > 0) s.half + m else s.half - 1 - m;
            const g = @as(f64, @floatFromInt(u)) + 0.5 - @as(f64, @floatFromInt(s.half));
            g2 += g * g;
            gy += g * v;
            sum += @floatFromInt(u);
            var b: usize = 0;
            while ((planes.len / words) > b) : (b += 1) {
                if ((u >> @intCast(b)) & 1 == 1) planes[b * words + i / 64] |= @as(u64, 1) << @intCast(i % 64);
            }
        }
        const gnorm = @sqrt(g2);
        return .{ .dot = gy / gnorm, .gnorm = gnorm, .sum = sum };
    }
};

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
    if (ix.centroid.len != ix.dims or ix.padded < ix.dims) return error.DimensionMismatch;
    if (ix.words != ix.padded / 64 + @intFromBool(ix.padded % 64 != 0)) return error.DimensionMismatch;
    const y = try gpa.alloc(f64, ix.padded);
    errdefer gpa.free(y);
    const norm = @sqrt(centre(y, q, ix.centroid));
    try rotate.applyInPlace(ix.seed, y);
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

/// Checks that `ix` is consistent and that `q` was prepared for an index of its shape.
fn checkPair(ix: Index, q: Query) Error!void {
    if (ix.bits == 0 or ix.bits > 8) return error.CodeBitsOutOfRange;
    if (q.bq > 8) return error.QueryBitsOutOfRange;
    if (ix.padded < 2 or ix.words != ix.padded / 64 + @intFromBool(ix.padded % 64 != 0)) return error.DimensionMismatch;
    if (ix.norms.len != ix.n or ix.dots.len != ix.n) return error.DimensionMismatch;
    if (ix.codes.len != try mulSize(&.{ ix.n, ix.bits, ix.words })) return error.DimensionMismatch;
    if (ix.bits > 1 and (ix.gnorms.len != ix.n or ix.sums.len != ix.n)) return error.DimensionMismatch;
    if (q.y.len != ix.padded or q.planes.len != @as(usize, q.bq) * ix.words) return error.DimensionMismatch;
}

/// The estimate for stored vector `i`; `error.IndexOutOfRange` if `i >= ix.n`, and
/// `error.DimensionMismatch` if `q` was prepared for an index of another shape.
pub fn estimate(ix: Index, q: Query, i: usize, eps0: f64) Error!Estimate {
    try checkPair(ix, q);
    if (i >= ix.n) return error.IndexOutOfRange;
    return estimateChecked(ix, q, i, eps0);
}

fn estimateChecked(ix: Index, q: Query, i: usize, eps0: f64) Estimate {
    const d: f64 = @floatFromInt(ix.padded);
    const norm = ix.norms[i];
    if (norm == 0 or q.norm == 0) return .{ .ip = 0, .bound = 0, .dist2 = norm * norm + q.norm * q.norm };
    const cw = ix.code(i);
    if (ix.bits > 1) return estimateEx(ix, q, i, cw, eps0);
    var on_sum: f64 = 0;
    var all_sum: f64 = 0;
    if (q.bq == 0) {
        for (cw, 0..) |word, wi| {
            var w = word;
            while (w != 0) : (w &= w - 1) {
                const j = wi * 64 + @ctz(w);
                if (j >= q.y.len) break;
                on_sum += q.y[j];
            }
        }
        for (q.y) |v| all_sum += v;
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

fn estimateEx(ix: Index, q: Query, i: usize, cw: []const u64, eps0: f64) Estimate {
    const d: f64 = @floatFromInt(ix.padded);
    const half: f64 = @floatFromInt(@as(u32, 1) << @intCast(ix.bits - 1));
    const shift = 0.5 - half;
    var s: f64 = 0;
    if (q.bq == 0) {
        for (q.y, 0..) |v, j| {
            var u: u32 = 0;
            for (0..ix.bits) |b| u |= @as(u32, @intCast((cw[b * ix.words + j / 64] >> @intCast(j % 64)) & 1)) << @intCast(b);
            s += (@as(f64, @floatFromInt(u)) + shift) * v;
        }
    } else {
        var uu: f64 = 0;
        for (0..ix.bits) |a| for (0..q.bq) |b| {
            var c: u64 = 0;
            for (cw[a * ix.words ..][0..ix.words], q.planes[b * ix.words ..][0..ix.words]) |w, p| c += @popCount(w & p);
            uu += @as(f64, @floatFromInt(c)) * @as(f64, @floatFromInt(@as(u32, 1) << @intCast(a + b)));
        };
        s = q.delta * uu + q.delta * shift * q.sum_u + q.low * ix.sums[i] + q.low * shift * d;
    }
    const dot = ix.dots[i];
    const ip = s / ix.gnorms[i] / dot;
    const bound = @sqrt(@max(0, 1 - dot * dot) / (dot * dot)) * eps0 / @sqrt(d - 1);
    const norm = ix.norms[i];
    return .{ .ip = ip, .bound = bound, .dist2 = norm * norm + q.norm * q.norm - 2 * norm * q.norm * ip };
}

pub const Hit = struct {
    id: u32,
    estimate: Estimate,
};

fn hitBefore(_: void, a: Hit, b: Hit) bool {
    return a.estimate.dist2 < b.estimate.dist2 or (a.estimate.dist2 == b.estimate.dist2 and a.id < b.id);
}

fn siftUpHitMax(heap: []Hit, idx_in: usize) void {
    var i = idx_in;
    while (i > 0) {
        const parent = (i - 1) / 2;
        if (!hitBefore({}, heap[parent], heap[i])) break;
        std.mem.swap(Hit, &heap[parent], &heap[i]);
        i = parent;
    }
}

fn siftDownHitMax(heap: []Hit, idx_in: usize) void {
    var i = idx_in;
    const n = heap.len;
    while (true) {
        var largest = i;
        const left = 2 * i + 1;
        const right = 2 * i + 2;
        if (left < n and hitBefore({}, heap[largest], heap[left])) largest = left;
        if (right < n and hitBefore({}, heap[largest], heap[right])) largest = right;
        if (largest == i) break;
        std.mem.swap(Hit, &heap[i], &heap[largest]);
        i = largest;
    }
}

/// Writes the `out.len` items with the smallest estimated distance into `out` (with their inner-product
/// estimates and confidence bounds), nearest first; ties go to the lower index. Zero allocations.
pub fn searchHits(ix: Index, q: Query, eps0: f64, out: []Hit) Error!usize {
    try checkPair(ix, q);
    if (ix.n > std.math.maxInt(u32)) return error.TooManyVectors;
    const k = @min(out.len, ix.n);
    if (k == 0) return 0;
    var count: usize = 0;
    for (0..ix.n) |i| {
        const cand: Hit = .{ .id = @intCast(i), .estimate = estimateChecked(ix, q, i, eps0) };
        if (count < k) {
            out[count] = cand;
            siftUpHitMax(out[0 .. count + 1], count);
            count += 1;
        } else if (hitBefore({}, cand, out[0])) {
            out[0] = cand;
            siftDownHitMax(out[0..k], 0);
        }
    }
    std.mem.sort(Hit, out[0..count], {}, hitBefore);
    return count;
}

/// The `out.len` indices with the smallest estimated distance, nearest first; ties to the lower index.
pub fn search(gpa: std.mem.Allocator, ix: Index, q: Query, out: []u32) Error!usize {
    const k = @min(out.len, ix.n);
    if (k == 0) return 0;
    const hits = try gpa.alloc(Hit, k);
    defer gpa.free(hits);
    const got = try searchHits(ix, q, default_eps0, hits);
    for (out[0..got], hits[0..got]) |*o, h| o.* = h.id;
    return got;
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
    try std.testing.expectApproxEqAbs(@as(f64, 1), (try estimate(ix, q, 0, default_eps0)).ip, 1e-12);
    const q4 = try prepare(gpa, ix, data[d .. 2 * d], 4, 99);
    defer q4.deinit(gpa);
    const qx = try prepare(gpa, ix, data[d .. 2 * d], 0, 0);
    defer qx.deinit(gpa);
    for (0..ix.n) |i| {
        const a = (try estimate(ix, q4, i, default_eps0)).ip;
        const b = (try estimate(ix, qx, i, default_eps0)).ip;
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
            const e = try estimate(ix, q, i, 1);
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

test "extended codes: tighter with more bits, still unbiased" {
    const gpa = std.testing.allocator;
    const d = 256;
    const data = try randomData(gpa, 300, d, 21);
    defer gpa.free(data);
    var prev_dot: f64 = 0;
    for ([_]u8{ 1, 2, 4, 8 }) |bits| {
        const ix = try buildBits(gpa, data, d, 3, null, bits);
        defer ix.deinit(gpa);
        var mean_dot: f64 = 0;
        for (ix.dots) |v| mean_dot += v;
        mean_dot /= @floatFromInt(ix.n);
        try std.testing.expect(mean_dot > prev_dot);
        prev_dot = mean_dot;
        const q = try prepare(gpa, ix, data[5 * d ..][0..d], 0, 0);
        defer q.deinit(gpa);
        try std.testing.expectApproxEqAbs(@as(f64, 1), (try estimate(ix, q, 5, default_eps0)).ip, 1e-12);
        var qn: f64 = 0;
        const qv = data[299 * d ..][0..d];
        for (qv, ix.centroid) |x, m| qn += (@as(f64, x) - m) * (@as(f64, x) - m);
        const qq = try prepare(gpa, ix, qv, 0, 0);
        defer qq.deinit(gpa);
        const q4 = try prepare(gpa, ix, qv, 4, 7);
        defer q4.deinit(gpa);
        var bias: f64 = 0;
        var sq: f64 = 0;
        var sigma: f64 = 0;
        for (0..ix.n - 1) |i| {
            var dot: f64 = 0;
            for (data[i * d ..][0..d], qv, ix.centroid) |a, b, m| dot += (@as(f64, a) - m) * (@as(f64, b) - m);
            const truth = dot / (ix.norms[i] * @sqrt(qn));
            const e = try estimate(ix, qq, i, 1);
            bias += e.ip - truth;
            sq += (e.ip - truth) * (e.ip - truth);
            sigma += e.bound * e.bound;
            // The 4-bit query path stays close to the exact query path.
            try std.testing.expect(@abs((try estimate(ix, q4, i, 1)).ip - e.ip) < 0.05);
        }
        const nt: f64 = @floatFromInt(ix.n - 1);
        try std.testing.expect(@abs(bias / nt) < 0.01);
        try std.testing.expect(@sqrt(sq / nt) < 1.3 * @sqrt(sigma / nt));
    }
    try std.testing.expect(prev_dot > 0.99);
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
    var hits: [5]Hit = undefined;
    try std.testing.expectEqual(@as(usize, 5), try searchHits(ix, q, default_eps0, &hits));
    for (out, hits) |id, h| {
        try std.testing.expectEqual(id, h.id);
        try std.testing.expect(h.estimate.bound >= 0);
    }
}

test "rejects bad input" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.DimensionMismatch, build(gpa, &.{ 1, 2, 3 }, 2, 0, null));
    try std.testing.expectError(error.EmptyIndex, build(gpa, &.{}, 2, 0, null));
    try std.testing.expectError(error.NonFiniteValue, build(gpa, &.{ 1, std.math.inf(f32) }, 2, 0, null));
}

test "estimate and searchHits reject out-of-range ids and mismatched queries" {
    const gpa = std.testing.allocator;
    const data = try randomData(gpa, 6, 10, 4);
    defer gpa.free(data);
    const ix = try buildBits(gpa, data, 10, 2, null, 3);
    defer ix.deinit(gpa);
    const q = try prepare(gpa, ix, data[0..10], 2, 1);
    defer q.deinit(gpa);
    _ = try estimate(ix, q, 5, default_eps0);
    try std.testing.expectError(error.IndexOutOfRange, estimate(ix, q, 6, default_eps0));
    try std.testing.expectError(error.IndexOutOfRange, estimate(ix, q, std.math.maxInt(usize), default_eps0));

    const wide_data = try randomData(gpa, 3, 40, 5);
    defer gpa.free(wide_data);
    const wide = try buildBits(gpa, wide_data, 40, 2, null, 3);
    defer wide.deinit(gpa);
    const wq = try prepare(gpa, wide, wide_data[0..40], 2, 1);
    defer wq.deinit(gpa);
    try std.testing.expectError(error.DimensionMismatch, estimate(ix, wq, 0, default_eps0));
    var hits: [2]Hit = undefined;
    try std.testing.expectError(error.DimensionMismatch, searchHits(ix, wq, default_eps0, &hits));
    var bad = q;
    bad.bq = 9;
    try std.testing.expectError(error.QueryBitsOutOfRange, estimate(ix, bad, 0, default_eps0));
    var bad_ix = ix;
    bad_ix.n = 7;
    try std.testing.expectError(error.DimensionMismatch, estimate(bad_ix, q, 6, default_eps0));
}

test "buildBits rejects dimensions whose padding overflows" {
    const gpa = std.testing.allocator;
    const dims = (@as(usize, 1) << 63) + 1;
    const fake: [*]const f32 = @ptrFromInt(@alignOf(f32));
    try std.testing.expectError(error.DimensionTooLarge, buildBits(gpa, fake[0..dims], dims, 0, null, 1));
}

test "centring huge values does not overflow" {
    const gpa = std.testing.allocator;
    const big = std.math.floatMax(f32);
    const data = [_]f32{ big, -big, -big, big };
    const ix = try build(gpa, &data, 2, 0, null);
    defer ix.deinit(gpa);
    for (ix.norms) |v| try std.testing.expect(std.math.isFinite(v) and v > 0);
    const q = try prepare(gpa, ix, data[0..2], 0, 0);
    defer q.deinit(gpa);
    try std.testing.expect(std.math.isFinite(q.norm));
    try std.testing.expectApproxEqAbs(@as(f64, 1), (try estimate(ix, q, 0, default_eps0)).ip, 1e-12);
}

test "one-bit exact-query estimate is bit-identical to a per-coordinate scan" {
    const gpa = std.testing.allocator;
    for ([_]usize{ 1, 3, 64, 65, 200 }) |d| {
        const data = try randomData(gpa, 20, d, d);
        defer gpa.free(data);
        const ix = try build(gpa, data, d, 9, null);
        defer ix.deinit(gpa);
        const q = try prepare(gpa, ix, data[d..][0..d], 0, 0);
        defer q.deinit(gpa);
        const pd: f64 = @floatFromInt(ix.padded);
        for (0..ix.n) |i| {
            if (ix.norms[i] == 0) continue;
            const cw = ix.code(i);
            var on_sum: f64 = 0;
            var all_sum: f64 = 0;
            for (q.y, 0..) |v, j| {
                if ((cw[j / 64] >> @intCast(j % 64)) & 1 == 1) on_sum += v;
                all_sum += v;
            }
            const ip = (2 * on_sum - all_sum) / @sqrt(pd) / ix.dots[i];
            try std.testing.expectEqual(ip, (try estimate(ix, q, i, default_eps0)).ip);
        }
    }
}

fn buildAlloc(gpa: std.mem.Allocator, data: []const f32, bits: u8) !void {
    const ix = try buildBits(gpa, data, 5, 3, null, bits);
    defer ix.deinit(gpa);
    const q = try prepare(gpa, ix, data[0..5], 4, 1);
    defer q.deinit(gpa);
    var out: [3]u32 = undefined;
    _ = try search(gpa, ix, q, &out);
}

test "buildBits, prepare and search free everything on allocation failure" {
    var data: [8 * 5]f32 = undefined;
    for (&data, 0..) |*x, i| x.* = @floatFromInt((i * 5) % 9);
    for ([_]u8{ 1, 2, 8 }) |bits| {
        try std.testing.checkAllAllocationFailures(std.testing.allocator, buildAlloc, .{ @as([]const f32, &data), bits });
    }
}
