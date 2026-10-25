//! Fuzz targets checking invariants that must hold for every input. `zig build test` runs every
//! input in `src/fuzz-corpus/`; `zig build test --fuzz` explores further.

const std = @import("std");
const index = @import("index.zig");
const Smith = std.testing.Smith;

const corpus = [_][]const u8{
    @embedFile("fuzz-corpus/00"),
    @embedFile("fuzz-corpus/01"),
    @embedFile("fuzz-corpus/02"),
    @embedFile("fuzz-corpus/03"),
};

fn upTo(smith: *Smith, lo: usize, hi: usize) usize {
    return smith.valueRangeAtMost(u32, @intCast(lo), @intCast(hi));
}

fn invariants(_: void, smith: *Smith) anyerror!void {
    const gpa = std.testing.allocator;
    const n = upTo(smith, 1, 24);
    const d = upTo(smith, 1, 40);
    var data: [24 * 40]f32 = undefined;
    for (data[0 .. n * d]) |*x| x.* = @as(f32, @floatFromInt(smith.value(i16))) / 1024;
    const bits: u8 = @intCast(upTo(smith, 1, 8));
    const ix = try index.buildBits(gpa, data[0 .. n * d], d, smith.value(u64), null, bits);
    defer ix.deinit(gpa);
    for (0..n) |i| {
        try std.testing.expect(ix.dots[i] > 0 and ix.dots[i] <= 1 + 1e-12);
        if (ix.norms[i] == 0) continue;
        const q = try index.prepare(gpa, ix, data[i * d ..][0..d], 0, 0);
        defer q.deinit(gpa);
        const own = try index.estimate(ix, q, i, index.default_eps0);
        try std.testing.expectApproxEqAbs(@as(f64, 1), own.ip, 1e-9);
        for (0..n) |j| {
            const e = try index.estimate(ix, q, j, index.default_eps0);
            try std.testing.expect(std.math.isFinite(e.ip) and e.bound >= 0 and std.math.isFinite(e.dist2));
        }
    }
}

test "fuzz: codes, dots and estimates stay well-formed" {
    try std.testing.fuzz({}, invariants, .{ .corpus = &corpus });
}
