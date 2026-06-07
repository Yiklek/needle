#include "gtest/gtest.h"

import fineflow.core.blob_tensor;
import fineflow.core.tensor;
import fineflow.core.op_kernel;
import fineflow.core.op_kernel_factory;
import fineflow.core.common.data_type_proto;
import fineflow.core.common.device_type_proto;
import fineflow.core.common.registry_manager;
import fineflow.core.kernels.dsl.dsl_kernel;
import fineflow.core.kernels.dsl.device_launcher;
import fineflow.core.kernels.dsl.dsl_registry;

using namespace fineflow;
using namespace fineflow::dsl;

// Test 1: DSLKernelMeta can be constructed with CPU compute function
TEST(DSLKernelMeta, ConstructWithCpuCompute) {
  DSLKernelMeta meta;
  meta.name = "test_add";
  meta.source_type = Source::kTileLang;
  meta.target_device = DeviceType::kCPU;

  meta.cpu_compute = [](KernelComputeContext& ctx) {
    auto in0 = *ctx.fetchTensor("in", 0);
    auto in1 = *ctx.fetchTensor("in", 1);
    auto out = *ctx.fetchTensor("out", 0);
    auto size = out.elementCount();
    auto* out_ptr = out.castPtrMut<float>();
    auto* in0_ptr = in0.castPtr<float>();
    auto* in1_ptr = in1.castPtr<float>();
    for (size_t i = 0; i < size; i++) {
      out_ptr[i] = in0_ptr[i] + in1_ptr[i];
    }
  };

  EXPECT_EQ(meta.name, "test_add");
  EXPECT_EQ(meta.source_type, Source::kTileLang);
  EXPECT_EQ(meta.target_device, DeviceType::kCPU);
  EXPECT_TRUE(static_cast<bool>(meta.cpu_compute));
}

// Test 2: CpuDeviceLauncher invokes the compute function
TEST(CpuDeviceLauncher, InvokesCompute) {
  auto a = CpuTensor::New(DataType::kFloat, Shape{4});
  auto b = CpuTensor::New(DataType::kFloat, Shape{4});
  auto c = CpuTensor::New(DataType::kFloat, Shape{4});

  a->castPtrMut<float>()[0] = 1.0f;
  a->castPtrMut<float>()[1] = 2.0f;
  a->castPtrMut<float>()[2] = 3.0f;
  a->castPtrMut<float>()[3] = 4.0f;
  b->castPtrMut<float>()[0] = 10.0f;
  b->castPtrMut<float>()[1] = 20.0f;
  b->castPtrMut<float>()[2] = 30.0f;
  b->castPtrMut<float>()[3] = 40.0f;

  DSLKernelMeta meta;
  meta.name = "add";
  meta.target_device = DeviceType::kCPU;
  meta.cpu_compute = [](KernelComputeContext& ctx) {
    auto in0 = *ctx.fetchTensor("in", 0);
    auto in1 = *ctx.fetchTensor("in", 1);
    auto out = *ctx.fetchTensor("out", 0);
    auto size = out.elementCount();
    auto* out_ptr = out.castPtrMut<float>();
    auto* in0_ptr = in0.castPtr<float>();
    auto* in1_ptr = in1.castPtr<float>();
    for (size_t i = 0; i < size; i++) {
      out_ptr[i] = in0_ptr[i] + in1_ptr[i];
    }
  };

  CpuDeviceLauncher launcher;
  KernelComputeContext ctx(DeviceType::kCPU, DataType::kFloat);
  ctx.insertTensor("in", 0, a->view());
  ctx.insertTensor("in", 1, b->view());
  ctx.insertTensor("out", 0, c->view());

  launcher.launch(meta, ctx);

  EXPECT_FLOAT_EQ(c->castPtr<float>()[0], 11.0f);
  EXPECT_FLOAT_EQ(c->castPtr<float>()[1], 22.0f);
  EXPECT_FLOAT_EQ(c->castPtr<float>()[2], 33.0f);
  EXPECT_FLOAT_EQ(c->castPtr<float>()[3], 44.0f);
}

// Test 3: DSLOpKernel wraps meta + launcher and delegates compute
TEST(DSLOpKernel, ComputeDelegatesToLauncher) {
  auto a = CpuTensor::New(DataType::kFloat, Shape{3});
  auto b = CpuTensor::New(DataType::kFloat, Shape{3});
  auto c = CpuTensor::New(DataType::kFloat, Shape{3});

  a->castPtrMut<float>()[0] = 1.0f;
  a->castPtrMut<float>()[1] = 2.0f;
  a->castPtrMut<float>()[2] = 3.0f;
  b->castPtrMut<float>()[0] = 5.0f;
  b->castPtrMut<float>()[1] = 5.0f;
  b->castPtrMut<float>()[2] = 5.0f;

  DSLKernelMeta meta;
  meta.name = "add";
  meta.target_device = DeviceType::kCPU;
  meta.cpu_compute = [](KernelComputeContext& ctx) {
    auto in0 = *ctx.fetchTensor("in", 0);
    auto in1 = *ctx.fetchTensor("in", 1);
    auto out = *ctx.fetchTensor("out", 0);
    auto size = out.elementCount();
    auto* out_ptr = out.castPtrMut<float>();
    auto* in0_ptr = in0.castPtr<float>();
    auto* in1_ptr = in1.castPtr<float>();
    for (size_t i = 0; i < size; i++) {
      out_ptr[i] = in0_ptr[i] + in1_ptr[i];
    }
  };

  auto launcher = DeviceLauncher::New(DeviceType::kCPU);
  ASSERT_TRUE(launcher.has_value());
  DSLOpKernel kernel(std::move(meta), std::move(*launcher));

  KernelComputeContext ctx(DeviceType::kCPU, DataType::kFloat);
  ctx.insertTensor("in", 0, a->view());
  ctx.insertTensor("in", 1, b->view());
  ctx.insertTensor("out", 0, c->view());

  kernel.compute(ctx);

  EXPECT_FLOAT_EQ(c->castPtr<float>()[0], 6.0f);
  EXPECT_FLOAT_EQ(c->castPtr<float>()[1], 7.0f);
  EXPECT_FLOAT_EQ(c->castPtr<float>()[2], 8.0f);
}

// Test 4: DSLOpKernelFactory creates kernel from meta
TEST(DSLOpKernelFactory, CreateReturnsValidKernel) {
  DSLKernelMeta meta;
  meta.name = "add";
  meta.target_device = DeviceType::kCPU;
  meta.cpu_compute = [](KernelComputeContext&) {};

  DSLOpKernelFactory factory(std::move(meta));
  auto kernel = factory.create(DataType::kFloat);

  EXPECT_TRUE(kernel.has_value());
  EXPECT_NE(kernel->get(), nullptr);
}

// Test 5: DSL kernel registers into RuntimeKernelFactoryRegistryMgr
TEST(DSLKernelRegistry, RegisterIntoRuntimeRegistry) {
  DSLKernelMeta meta;
  meta.name = "dsl_test_op";
  meta.target_device = DeviceType::kCPU;
  meta.cpu_compute = [](KernelComputeContext& ctx) {
    auto in0 = *ctx.fetchTensor("in", 0);
    auto out = *ctx.fetchTensor("out", 0);
    auto* out_ptr = out.castPtrMut<float>();
    auto* in0_ptr = in0.castPtr<float>();
    out_ptr[0] = in0_ptr[0] * 2.0f;
  };

  auto& registry = RuntimeKernelFactoryRegistryMgr::Get();
  EXPECT_FALSE(registry.IsRegistered({"dsl_test_op", DeviceType::kCPU}));

  auto reg_result = DSLKernelRegistry::Register(std::move(meta));
  EXPECT_TRUE(reg_result.has_value());

  EXPECT_TRUE(registry.IsRegistered({"dsl_test_op", DeviceType::kCPU}));
}

// Test 6: End-to-end: register, lookup, create, compute
TEST(DSLKernelE2E, RegisterLookupCreateCompute) {
  DSLKernelMeta meta;
  meta.name = "dsl_double";
  meta.target_device = DeviceType::kCPU;
  meta.cpu_compute = [](KernelComputeContext& ctx) {
    auto in0 = *ctx.fetchTensor("in", 0);
    auto out = *ctx.fetchTensor("out", 0);
    out.castPtrMut<float>()[0] = in0.castPtr<float>()[0] * 2.0f;
  };

  auto reg_result = DSLKernelRegistry::Register(std::move(meta));
  EXPECT_TRUE(reg_result.has_value());

  auto a = CpuTensor::New(DataType::kFloat);
  auto c = CpuTensor::New(DataType::kFloat);
  a->castPtrMut<float>()[0] = 21.0f;

  auto factory_result = RuntimeKernelFactoryRegistryMgr::Get().GetValue({"dsl_double", DeviceType::kCPU});
  EXPECT_TRUE(factory_result.has_value());

  auto kernel_result = (**factory_result)->create(DataType::kFloat);
  EXPECT_TRUE(kernel_result.has_value());

  KernelComputeContext ctx(DeviceType::kCPU, DataType::kFloat);
  ctx.insertTensor("in", 0, a->view());
  ctx.insertTensor("out", 0, c->view());

  (*kernel_result)->compute(ctx);

  EXPECT_FLOAT_EQ(c->castPtr<float>()[0], 42.0f);
}

// Test 7: KernelComputeContext attrs round-trip
TEST(KernelComputeContext, AttrsRoundTrip) {
  KernelComputeContext ctx(DeviceType::kCPU, DataType::kFloat);
  AttrMap attrs;
  attrs["strides"] = std::vector<int64_t>{1, 1};
  attrs["pads"] = std::vector<int64_t>{0, 0, 0, 0};
  attrs["group"] = int64_t(1);
  attrs["alpha"] = 0.5;
  attrs["auto_pad"] = std::string("SAME_UPPER");
  ctx.setAttrs(std::move(attrs));
  EXPECT_EQ(std::get<std::vector<int64_t>>(ctx.attrs().at("strides")),
            (std::vector<int64_t>{1, 1}));
  EXPECT_EQ(std::get<int64_t>(ctx.attrs().at("group")), 1);
  EXPECT_DOUBLE_EQ(std::get<double>(ctx.attrs().at("alpha")), 0.5);
  EXPECT_EQ(std::get<std::string>(ctx.attrs().at("auto_pad")), "SAME_UPPER");
}

// Test 8: DSLOpKernel compute receives attrs
TEST(DSLOpKernel, ComputeReceivesAttrs) {
  auto a = CpuTensor::New(DataType::kFloat, Shape{4});
  auto c = CpuTensor::New(DataType::kFloat, Shape{4});
  a->castPtrMut<float>()[0] = 3.0f;
  DSLKernelMeta meta;
  meta.name = "double_with_alpha";
  meta.target_device = DeviceType::kCPU;
  meta.attrs_schema["alpha"] = double(2.0);
  meta.cpu_compute = [](KernelComputeContext& ctx) {
    double alpha = std::get<double>(ctx.attrs().at("alpha"));
    auto in0 = *ctx.fetchTensor("in", 0);
    auto out = *ctx.fetchTensor("out", 0);
    out.castPtrMut<float>()[0] = in0.castPtr<float>()[0] * static_cast<float>(alpha);
  };
  CpuDeviceLauncher launcher;
  KernelComputeContext ctx(DeviceType::kCPU, DataType::kFloat);
  ctx.setAttrs(meta.attrs_schema);
  ctx.insertTensor("in", 0, a->view());
  ctx.insertTensor("out", 0, c->view());
  launcher.launch(meta, ctx);
  EXPECT_FLOAT_EQ(c->castPtr<float>()[0], 6.0f);
}
