#include "gtest/gtest.h"

import fineflow.core.blob_tensor;
import fineflow.core.common.data_type_proto;

using namespace fineflow;
BlobTensorView getView() {
  auto t = CpuTensor::New(DataType::kFloat, {2, 3});
  auto r = CloneTensor(t->view());
  return r;
}

TEST(Tensor, UseCount) {
  auto v = getView();
  EXPECT_EQ(1, v.ptr().use_count());
  auto v2 = v;
  EXPECT_EQ(2, v.ptr().use_count());
  EXPECT_EQ(2, v2.ptr().use_count());
}

TEST(Tensor, Clone) {
  auto v = getView();
  auto v2 = CloneTensor(v);
  EXPECT_NE(v.ptr().get(), v2.ptr().get());
  EXPECT_NE(v.rawPtr(), v2.rawPtr());
  EXPECT_EQ(v.bufferSize(), v2.bufferSize());
  EXPECT_EQ(v.shape(), v2.shape());
  EXPECT_EQ(v.stride(), v2.stride());
  EXPECT_EQ(v.isScalar(), v2.isScalar());
}
