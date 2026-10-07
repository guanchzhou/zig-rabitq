const std = @import("std");
const rabitq = @import("rabitq");
const index = rabitq.index;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var prng = std.Random.DefaultPrng.init(42);
    const r = prng.random();
    const n = 500;
    const d = 128;
    const data = try gpa.alloc(f32, n * d);
    defer gpa.free(data);
    for (data) |*x| x.* = r.floatNorm(f32);

    // One bit per dimension: 128 bits per vector instead of 128 floats.
    const ix = try index.build(gpa, data, d, 0x9e3779b97f4a7c15, null);
    defer ix.deinit(gpa);

    // A query close to vector 123: the 4-bit query scan should put it first.
    var query: [d]f32 = undefined;
    for (&query, data[123 * d ..][0..d]) |*q, x| q.* = x + 0.1 * r.floatNorm(f32);
    const q = try index.prepare(gpa, ix, &query, 4, 1);
    defer q.deinit(gpa);
    var top: [10]u32 = undefined;
    _ = try index.search(gpa, ix, q, &top);
    std.debug.assert(top[0] == 123);

    // The error bound brackets the true inner product for the found vector.
    const e = index.estimate(ix, q, 123, 3);
    std.debug.assert(e.ip - e.bound < 1 and e.bound > 0);
}
