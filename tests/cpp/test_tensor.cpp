#include "catch2/catch_test_macros.hpp"
#include "fineflow/core/cpu/cpu_tensor.h"
#include "fineflow/core/tensor_util.h"

using namespace fineflow;
BlobTensorView getView() {
  auto t = CpuTensor::New(DataType::kFloat, {2, 3});
  auto r = CloneTensor(t->view());
  return r;
}

TEST_CASE("tensor") {
  auto v = getView();
  REQUIRE(1 == v.ptr().use_count());
}

TEST_CASE("tensor clone") {
  auto v = getView();
  auto v2 = CloneTensor(v);
  REQUIRE(v.ptr().get() != v2.ptr().get());
  REQUIRE(v.rawPtr() != v2.rawPtr());
  REQUIRE(v.bufferSize() == v2.bufferSize());
  REQUIRE(v.shape() == v2.shape());
  REQUIRE(v.stride() == v2.stride());
  REQUIRE(v.isScalar() == v2.isScalar());
}
