#ifndef FINEFLOW_CORE_CPU_CPU_TENSOR_H_
#define FINEFLOW_CORE_CPU_CPU_TENSOR_H_
#include <cstdlib>

#include "fineflow/core/blob_tensor.h"
#include "fmt/format.h"

namespace fineflow {

class CpuTensor final : public AllocableBlobTensor<> {
public:
  using AllocableBlobTensor::AllocableBlobTensor;
  CpuTensor(DataType dtype, uint64_t buffer_size) : AllocableBlobTensor<>(DeviceType::kCPU, dtype, buffer_size) {}
  CpuTensor(DataType dtype, const Shape& shape) : AllocableBlobTensor<>(DeviceType::kCPU, dtype, shape) {}
};
}  // namespace fineflow
#endif  // FINEFLOW_CORE_CPU_CPU_TENSOR_H_
