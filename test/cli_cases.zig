//! Command-line cases run by `zig build test-cli`. `stdout` is compared
//! exactly, `stderr` as a substring.

pub const Case = struct {
    name: []const u8,
    args: []const []const u8,
    stdin: ?[]const u8 = null,
    stdout: ?[]const u8 = null,
    stderr: ?[]const u8 = null,
    exit: u8 = 0,
};

pub const cases = [_]Case{
    .{
        .name = "search: estimates then exact re-rank",
        .args = &.{"search"},
        .stdin = "{\"data\":[[1,0,0],[0.9,0.1,0],[0,1,0],[0,0,1],[-1,0,0]],\"queries\":[[1,0.05,0]],\"k\":3,\"rerank\":5}",
        .stdout = "{\"results\":[[0,1,2]]}\n",
    },
    .{
        .name = "encode: codes, norms and dots",
        .args = &.{"encode"},
        .stdin = "{\"data\":[[1,0,0],[0,1,0]]}",
        .stdout = "{\"padded\":4,\"codes\":[\"000000000000000a\",\"0000000000000000\"],\"norms\":[0.7071067811865476,0.7071067811865476],\"dots\":[0.7071067811865475,0.7071067811865475]}\n",
    },
    .{ .name = "error: ragged rows", .args = &.{"encode"}, .stdin = "{\"data\":[[1,2],[3]]}", .stderr = "data row 1 has 1 values, expected 2", .exit = 1 },
    .{ .name = "error: query width", .args = &.{"search"}, .stdin = "{\"data\":[[1,2]],\"queries\":[[1]]}", .stderr = "queries have 1 values, data has 2", .exit = 1 },
    .{ .name = "error: truncated JSON", .args = &.{"search"}, .stdin = "{", .stderr = "invalid input", .exit = 1 },
    .{ .name = "error: unknown command", .args = &.{"bogus"}, .stderr = "unknown command: bogus", .exit = 2 },
    .{ .name = "error: no command prints usage", .args = &.{}, .stderr = "usage: zig-rabitq COMMAND", .exit = 2 },
    .{ .name = "help: exits 0", .args = &.{"--help"} },
};
