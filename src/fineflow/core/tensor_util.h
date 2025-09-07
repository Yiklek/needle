#ifndef RELATIVE_FILEPATH
#define RELATIVE_FILEPATH
#include "fineflow/core/cpu/cpu_tensor.h"
namespace fineflow {

inline BlobTensorPtr DeriveEmptyTensorLike(const BlobTensorView& tensor) {
  if (tensor.device() == DeviceType::kCPU) {
    return CpuTensor::New(tensor.dtype(), tensor.shape());
  }
  return nullptr;
}

template <class T>
inline BlobTensorView DeriveScalarOnSameDevice(const BlobTensorView& tensor, T scalar) requires (is_cpu_native_v<T>) {
  if (tensor.device() == DeviceType::kCPU) {
    return CpuTensor::New(std::forward<T>(scalar));
  }
  return BlobTensorView(nullptr);
}
inline BlobTensorView CloneTensor(const BlobTensorView& tensor) {
  if (tensor.device() == DeviceType::kCPU) {
    return CpuTensor::Clone(tensor);
  }
  return BlobTensorView(nullptr);  // TODO(y): should not return null.
}

}  // namespace fineflow
#endif  // RELATIVE_FILEPATH
