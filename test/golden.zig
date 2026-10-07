//! Replays test/golden.json, the outputs of the independent reference in test/reference.py.
//! Codes and top-5 orders must match exactly; norms, dots and estimates within 1e-9.

const std = @import("std");
const rabitq = @import("rabitq");
const index = rabitq.index;

const Est = struct { ip: f64, bound: f64, dist2: f64 };
const Mode = struct { round_seed: []const u8, estimates: []const Est, top5: []const u32 };
const Golden = struct {
    cases: []const struct {
        d: usize,
        bits: u8,
        seed: []const u8,
        data: []const f32,
        padded: usize,
        codes: []const []const []const u8,
        norms: []const f64,
        dots: []const f64,
        queries: []const struct { q: []const f32, bq0: Mode, bq4: Mode },
    },
};

fn close(want: f64, got: f64) !void {
    if (!(@abs(want - got) <= 1e-9 * @max(1, @abs(want)))) {
        std.debug.print("want {d}, got {d}\n", .{ want, got });
        return error.TestExpectedApproxEq;
    }
}

test "codes, estimates and search match the NumPy reference" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const g = try std.json.parseFromSliceLeaky(Golden, arena, @embedFile("golden.json"), .{});
    for (g.cases) |c| {
        const seed = try std.fmt.parseInt(u64, c.seed, 10);
        const ix = try index.buildBits(arena, c.data, c.d, seed, null, c.bits);
        try std.testing.expectEqual(c.padded, ix.padded);
        for (c.codes, 0..) |words, i| {
            for (words, ix.code(i)) |hex, w| try std.testing.expectEqual(try std.fmt.parseInt(u64, hex, 16), w);
            try close(c.norms[i], ix.norms[i]);
            try close(c.dots[i], ix.dots[i]);
        }
        for (c.queries) |q| {
            inline for (.{ .{ "bq0", 0 }, .{ "bq4", 4 } }) |m| {
                const mode: Mode = @field(q, m[0]);
                const qq = try index.prepare(arena, ix, q.q, m[1], try std.fmt.parseInt(u64, mode.round_seed, 10));
                for (mode.estimates, 0..) |want, i| {
                    const e = index.estimate(ix, qq, i, index.default_eps0);
                    try close(want.ip, e.ip);
                    try close(want.bound, e.bound);
                    try close(want.dist2, e.dist2);
                }
                var top: [5]u32 = undefined;
                const k = try index.search(arena, ix, qq, &top);
                try std.testing.expectEqualSlices(u32, mode.top5, top[0..k]);
            }
        }
    }
}
