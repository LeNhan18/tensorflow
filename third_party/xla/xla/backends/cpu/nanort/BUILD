load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "nanort_client",
    srcs = ["nanort_client.cc"],
    hdrs = ["nanort_client.h"],
    deps = [
        ":nanort_executable",
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:utils",
        "//xla/service:compiler",
        "//xla/service:dump",
        "//xla/service:executable",
        "//xla/service:hlo_module_config",
        "//xla/service/cpu:cpu_compiler_pure",
        "//xla/service/cpu:cpu_executable",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/lib:traceme_encode",
    ],
)

cc_test(
    name = "nanort_client_test",
    srcs = ["nanort_client_test.cc"],
    deps = [
        ":nanort_client",
        ":nanort_executable",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/tsl/concurrency:async_value",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "nanort_executable",
    srcs = ["nanort_executable.cc"],
    hdrs = ["nanort_executable.h"],
    deps = [
        "//xla:util",
        "//xla/backends/cpu/runtime:buffer_allocations",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/service:executable",
        "//xla/service:maybe_owning_device_memory",
        "//xla/service/cpu:cpu_executable",
        "//xla/stream_executor:device_memory",
        "//xla/tsl/concurrency:async_value",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/lib:traceme_encode",
    ],
)
