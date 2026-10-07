//! zig-rabitq: RaBitQ binary codes for approximate nearest-neighbour search.
//!
//! - `rotate`: a seeded random orthogonal transform (random signs and Hadamard, three rounds).
//! - `index`: one-bit codes, the unbiased inner-product estimator and its error bound, query
//!   quantisation for popcount scans, and a top-k search by estimated distance.

const std = @import("std");

pub const rotate = @import("rotate.zig");
pub const index = @import("index.zig");

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(rotate);
    std.testing.refAllDecls(index);
    _ = @import("fuzz.zig");
}
