const std = @import("std");
const zon = @import("build.zig.zon");
const cli_cases = @import("test/cli_cases.zig");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const rabitq = b.addModule("rabitq", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const options = b.addOptions();
    options.addOption([]const u8, "version", zon.version);
    const cli = b.addExecutable(.{
        .name = "zig-rabitq",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "rabitq", .module = rabitq },
                .{ .name = "build_options", .module = options.createModule() },
            },
        }),
    });
    b.installArtifact(cli);
    const run_cli = b.addRunArtifact(cli);
    run_cli.addPassthruArgs();
    b.step("run", "Run the zig-rabitq command").dependOn(&run_cli.step);

    const test_step = b.step("test", "Run the test suite");
    const unit = b.addTest(.{ .root_module = rabitq });
    test_step.dependOn(&b.addRunArtifact(unit).step);

    const golden = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("test/golden.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "rabitq", .module = rabitq }},
    }) });
    test_step.dependOn(&b.addRunArtifact(golden).step);

    for ([_][]const u8{"search"}) |name| {
        const example = b.addExecutable(.{
            .name = name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("examples/{s}.zig", .{name})),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "rabitq", .module = rabitq }},
            }),
        });
        const run = b.addRunArtifact(example);
        run.expectExitCode(0);
        test_step.dependOn(&run.step);
    }

    const test_cli = b.step("test-cli", "Run the command-line tests");
    for (cli_cases.cases) |c| {
        const run = b.addRunArtifact(cli);
        run.setName(b.fmt("zig-rabitq {s}", .{c.name}));
        run.addArgs(c.args);
        run.setStdIn(if (c.stdin) |s| .{ .bytes = s } else .none);
        if (c.stdout) |s| run.expectStdOutEqual(s);
        if (c.stderr) |s| run.expectStdErrMatch(s);
        run.expectExitCode(c.exit);
        test_cli.dependOn(&run.step);
    }
    const version = b.addRunArtifact(cli);
    version.setName("zig-rabitq version");
    version.addArg("version");
    version.expectStdOutEqual(b.fmt("zig-rabitq {s}\n", .{zon.version}));
    test_cli.dependOn(&version.step);
    test_step.dependOn(test_cli);

    const docs = b.addInstallDirectory(.{
        .source_dir = unit.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    b.step("docs", "Write the API documentation to zig-out/docs").dependOn(&docs.step);
}
