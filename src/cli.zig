const std = @import("std");
const rabitq = @import("rabitq");
const build_options = @import("build_options");
const index = rabitq.index;

const usage =
    \\usage: zig-rabitq COMMAND < input.json
    \\
    \\  encode  {"data":[[..]..],"seed":"HEX"?}
    \\          -> {"padded":D,"codes":["HEX"..],"norms":[..],"dots":[..]}
    \\  search  {"data":[[..]..],"queries":[[..]..],"k":K,"bq":B?,"rerank":R?,"seed":"HEX"?}
    \\          -> {"results":[[i..]..]}   nearest first by estimated distance; with R > 0 the
    \\             R best estimates are re-ranked by exact distance before the top K are kept
    \\  version
    \\
;

const default_seed: u64 = 0x9e3779b97f4a7c15;

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("zig-rabitq: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}

fn parse(comptime T: type, arena: std.mem.Allocator, text: []const u8) T {
    return std.json.parseFromSliceLeaky(T, arena, text, .{ .ignore_unknown_fields = true }) catch |e|
        fail("invalid input: {s}", .{@errorName(e)});
}

fn emit(out: *std.Io.Writer, v: anytype) !void {
    try std.json.Stringify.value(v, .{}, out);
    try out.writeAll("\n");
}

fn flatten(arena: std.mem.Allocator, rows: []const []const f32, what: []const u8) !struct { data: []f32, d: usize } {
    if (rows.len == 0) fail("{s} needs at least one row", .{what});
    const d = rows[0].len;
    if (d == 0) fail("{s} rows must not be empty", .{what});
    const data = try arena.alloc(f32, rows.len * d);
    for (rows, 0..) |r, i| {
        if (r.len != d) fail("{s} row {d} has {d} values, expected {d}", .{ what, i, r.len, d });
        @memcpy(data[i * d ..][0..d], r);
    }
    return .{ .data = data, .d = d };
}

fn seedOf(text: ?[]const u8) u64 {
    const s = text orelse return default_seed;
    return std.fmt.parseInt(u64, s, 16) catch fail("seed must be hexadecimal: {s}", .{s});
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    var buf: [64 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &buf);
    const out = &stdout.interface;
    defer out.flush() catch {};
    if (args.len != 2) {
        std.debug.print("{s}", .{usage});
        std.process.exit(2);
    }
    const cmd: []const u8 = args[1];
    if (std.mem.eql(u8, cmd, "version") or std.mem.eql(u8, cmd, "--version")) {
        try out.print("zig-rabitq {s}\n", .{build_options.version});
        return;
    }
    if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        try out.writeAll(usage);
        return;
    }
    if (!std.mem.eql(u8, cmd, "encode") and !std.mem.eql(u8, cmd, "search")) {
        std.debug.print("zig-rabitq: unknown command: {s}\n{s}", .{ cmd, usage });
        std.process.exit(2);
    }
    var in_buf: [64 * 1024]u8 = undefined;
    var stdin = std.Io.File.stdin().readerStreaming(init.io, &in_buf);
    const text = stdin.interface.allocRemaining(arena, .unlimited) catch |e| switch (e) {
        error.ReadFailed => return stdin.err.?,
        else => |x| return x,
    };

    if (std.mem.eql(u8, cmd, "encode")) {
        const in = parse(struct { data: []const []const f32, seed: ?[]const u8 = null }, arena, text);
        const m = try flatten(arena, in.data, "data");
        const ix = index.build(arena, m.data, m.d, seedOf(in.seed), null) catch |e| fail("{s}", .{@errorName(e)});
        const codes = try arena.alloc([]const u8, ix.n);
        for (codes, 0..) |*c, i| {
            const hex = try arena.alloc(u8, ix.words * 16);
            for (ix.code(i), 0..) |w, j| _ = std.fmt.bufPrint(hex[j * 16 ..][0..16], "{x:0>16}", .{w}) catch unreachable;
            c.* = hex;
        }
        try emit(out, .{ .padded = ix.padded, .codes = codes, .norms = ix.norms, .dots = ix.dots });
        return;
    }

    const in = parse(struct {
        data: []const []const f32,
        queries: []const []const f32,
        k: usize = 10,
        bq: u8 = 4,
        rerank: usize = 0,
        seed: ?[]const u8 = null,
    }, arena, text);
    const m = try flatten(arena, in.data, "data");
    const qs = try flatten(arena, in.queries, "queries");
    if (qs.d != m.d) fail("queries have {d} values, data has {d}", .{ qs.d, m.d });
    if (in.k == 0) fail("k must be at least 1", .{});
    const ix = index.build(arena, m.data, m.d, seedOf(in.seed), null) catch |e| fail("{s}", .{@errorName(e)});
    const results = try arena.alloc([]u32, in.queries.len);
    const depth = @max(in.k, in.rerank);
    for (results, 0..) |*res, qi| {
        const qv = qs.data[qi * m.d ..][0..m.d];
        const q = index.prepare(arena, ix, qv, in.bq, qi) catch |e| fail("{s}", .{@errorName(e)});
        const cand = try arena.alloc(u32, @min(depth, ix.n));
        const got = try index.search(arena, ix, q, cand);
        if (in.rerank > 0) {
            const Item = struct { i: u32, d: f64 };
            const items = try arena.alloc(Item, got);
            for (items, cand[0..got]) |*it, i| {
                var s: f64 = 0;
                for (m.data[@as(usize, i) * m.d ..][0..m.d], qv) |a, b| s += (@as(f64, a) - b) * (@as(f64, a) - b);
                it.* = .{ .i = i, .d = s };
            }
            std.mem.sort(Item, items, {}, struct {
                fn f(_: void, a: Item, b: Item) bool {
                    return a.d < b.d or (a.d == b.d and a.i < b.i);
                }
            }.f);
            for (cand[0..got], items) |*c, it| c.* = it.i;
        }
        res.* = cand[0..@min(in.k, got)];
    }
    try emit(out, .{ .results = results });
}
