#ifndef FINEFLOW_API_PYTHON_PY_TENSOR_H_
#define FINEFLOW_API_PYTHON_PY_TENSOR_H_
#include <memory>

#include "fineflow/core/blob_tensor.h"
#include "fineflow/core/common/device_type.pb.h"
#include "fineflow/core/cpu/cpu_tensor.h"
namespace fineflow::python_api {

class Tensor final {
private:
  BlobTensorPtr tensor_;
  Tensor() = default;
  template <class T>
  explicit Tensor(T* tensor) : tensor_(tensor) {}

public:
  Tensor(const BlobTensorPtr& tensor) : tensor_(tensor) {}        // NOLINT
  Tensor(BlobTensorPtr&& tensor) : tensor_(std::move(tensor)) {}  // NOLINT
  static Tensor New(DeviceType device, uint64_t size, DataType dtype = DataType::kFloat) {
    if (device == kCPU) {
      return Tensor(new CpuTensor(dtype, size));
    }
    return Tensor();
  }
  const BlobTensorPtr& operator->() const { return tensor_; }
  BlobTensorPtr& operator->() { return tensor_; }

  const BlobTensorPtr& operator*() const { return tensor_; }
  BlobTensorPtr& operator*() { return tensor_; }

  [[nodiscard]] inline const BlobTensorPtr& ptr() const { return tensor_; }
  inline operator const BlobTensorPtr&() const { return tensor_; }  // NOLINT
  inline operator BlobTensorPtr&() { return tensor_; }              // NOLINT
  ~Tensor() {
    LOG(trace) << fmt::format("Deconstruct python_api tensor. Core tensor (Use count): {} ({}). Buffer: {}",
                              fmt::ptr(tensor_.get()), tensor_.use_count(), fmt::ptr(tensor_->rawPtr()));
  }
};

}  // namespace fineflow::python_api
#endif  // FINEFLOW_API_PYTHON_PY_TENSOR_H_
