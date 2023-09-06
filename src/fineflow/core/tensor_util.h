#ifndef RELATIVE_FILEPATH
#define RELATIVE_FILEPATH
#include "fineflow/core/cpu/cpu_tensor.h"
namespace fineflow {

inline BlobTensorPtr DeriveEmptyTensorLike(const BlobTensorPtr& tensor) {
  if (tensor->device() == DeviceType::kCPU) {
    return BlobTensorPtr(new CpuTensor(tensor->dtype(), tensor->shape()));
  }
  return nullptr;
}
inline BlobTensorPtr CopyTensorFrom(const BlobTensorPtr& tensor) {
  if (tensor->device() == DeviceType::kCPU) {
    auto* n = new CpuTensor(*static_cast<CpuTensor*>(tensor.get()));
    return BlobTensorPtr(n);
  }
  return nullptr;
}

}  // namespace fineflow
#endif  // RELATIVE_FILEPATH