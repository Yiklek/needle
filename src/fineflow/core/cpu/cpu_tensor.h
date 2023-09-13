#ifndef FINEFLOW_CORE_CPU_CPU_TENSOR_H_
#define FINEFLOW_CORE_CPU_CPU_TENSOR_H_
#include <cstdlib>

#include "fineflow/core/blob_tensor.h"
#include "fineflow/core/common/fmt.hpp"

namespace fineflow {

class CpuTensor final : public AllocableBlobTensor<> {
private:
  FF_DEFAULT_COPY_AND_MOVE(CpuTensor)
  CpuTensor(DataType dtype, uint64_t buffer_size) : AllocableBlobTensor(DeviceType::kCPU, dtype, buffer_size) {}
  CpuTensor(DataType dtype, const Shape& shape) : AllocableBlobTensor(DeviceType::kCPU, dtype, shape) {}
  explicit CpuTensor(DataType dtype) : AllocableBlobTensor(DeviceType::kCPU, dtype) {}

  template <class T, class Enable = std::enable_if_t<is_cpu_native_v<T>>>
  explicit CpuTensor(T&& t) : AllocableBlobTensor<>(DeviceType::kCPU, std::forward<T>(t)) {}

public:
  static BlobTensorPtr New(DataType dtype, uint64_t buffer_size) {
    return BlobTensorPtr(new CpuTensor(dtype, buffer_size));
  }
  static BlobTensorPtr New(DataType dtype, const Shape& shape) { return BlobTensorPtr(new CpuTensor(dtype, shape)); }
  static BlobTensorPtr New(DataType dtype) { return BlobTensorPtr(new CpuTensor(dtype)); }

  template <class T, class Enable = std::enable_if_t<is_cpu_native_v<T>>>
  static BlobTensorPtr New(T&& t) {
    auto* p = new CpuTensor(std::forward<T>(t));
    return BlobTensorPtr(p);
  }
  static BlobTensorPtr Clone(const BlobTensorPtr& other) {
    return BlobTensorPtr(new CpuTensor(*static_cast<CpuTensor*>(other.get())));
  }

  static BlobTensorView Clone(const BlobTensorView& other) {
    auto p = BlobTensorPtr(new CpuTensor(*static_cast<CpuTensor*>(other.ptr().get())));  // copy tensor
    return BlobTensorView(p);                                                            // copy view
  }
};
}  // namespace fineflow
#endif  // FINEFLOW_CORE_CPU_CPU_TENSOR_H_
