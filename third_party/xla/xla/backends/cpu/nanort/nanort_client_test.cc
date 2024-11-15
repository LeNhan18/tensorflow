/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/cpu/nanort/nanort_client.h"

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/nanort/nanort_executable.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::cpu {
namespace {

absl::StatusOr<XlaComputation> CreateAddScalarsComputation() {
  XlaBuilder b("add");

  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {}), "p1");
  Add(Add(p0, p1), Add(p0, p1));

  return b.Build();
}

absl::StatusOr<XlaComputation> CreateFibonacciComputation() {
  XlaBuilder b("fib");

  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {}), "p1");

  std::vector<XlaOp> vars = {p0, p1};

  static constexpr int kFibonacciNumber = 20;
  for (int i = 2; i < kFibonacciNumber; ++i) {
    vars.push_back(Add(vars[i - 2], vars[i - 1]));
  }

  return b.Build();
}

TEST(NanoRtClientTest, CompileAndRun) {
  NanoRtClient client;

  TF_ASSERT_OK_AND_ASSIGN(auto computation, CreateAddScalarsComputation());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<NanoRtExecutable> executable,
                          client.Compile(computation));

  // Storage for executable parameters and results.
  alignas(32) float p0_value = 1.0f;
  alignas(32) float p1_value = 2.0f;
  alignas(32) float result = 0.0f;

  // Prepare executable parameters, results and temp storage.
  NanoRtExecutable::Parameter p0(&p0_value, 1);
  NanoRtExecutable::Parameter p1(&p1_value, 1);
  NanoRtExecutable::Result r0(&result, 1);
  NanoRtExecutable::Temp temp = {};

  std::vector<NanoRtExecutable::Parameter> params = {p0, p1};
  std::vector<NanoRtExecutable::Result> results = {r0};

  auto event = executable->Execute(params, results, temp);
  tsl::BlockUntilReady(event);

  ASSERT_TRUE(event.IsConcrete());
  EXPECT_EQ(result, 6.0f);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

static void BM_AddScalars(benchmark::State& state) {
  NanoRtClient client;

  auto computation = CreateAddScalarsComputation();
  auto executable = client.Compile(*computation);

  // Storage for executable parameters and results.
  alignas(32) float p0_value = 1.0f;
  alignas(32) float p1_value = 2.0f;
  alignas(32) float result = 0.0f;

  // Prepare executable parameters, results and temp storage.
  NanoRtExecutable::Parameter p0(&p0_value, 1);
  NanoRtExecutable::Parameter p1(&p1_value, 1);
  NanoRtExecutable::Result r0(&result, 1);
  NanoRtExecutable::Temp temp = {};

  std::vector<NanoRtExecutable::Parameter> params = {p0, p1};
  std::vector<NanoRtExecutable::Result> results = {r0};

  for (auto _ : state) {
    auto event = (*executable)->Execute(params, results, temp);
    tsl::BlockUntilReady(event);
  }
}

BENCHMARK(BM_AddScalars);

static void BM_Fibonacci(benchmark::State& state) {
  NanoRtClient client;

  auto computation = CreateFibonacciComputation();
  auto executable = client.Compile(*computation);

  // Storage for executable parameters and results.
  alignas(32) float p0_value = 1.0f;
  alignas(32) float p1_value = 2.0f;
  alignas(32) float result = 0.0f;

  // Prepare executable parameters, results and temp storage.
  NanoRtExecutable::Parameter p0(&p0_value, 1);
  NanoRtExecutable::Parameter p1(&p1_value, 1);
  NanoRtExecutable::Result r0(&result, 1);
  NanoRtExecutable::Temp temp = {};

  std::vector<NanoRtExecutable::Parameter> params = {p0, p1};
  std::vector<NanoRtExecutable::Result> results = {r0};

  for (auto _ : state) {
    auto event = (*executable)->Execute(params, results, temp);
    tsl::BlockUntilReady(event);
  }
}

BENCHMARK(BM_Fibonacci);

}  // namespace
}  // namespace xla::cpu
