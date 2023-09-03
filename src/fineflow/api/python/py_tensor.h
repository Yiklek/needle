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
  explicit Tensor(BlobTensor* tensor) : tensor_(tensor) {}

public:
  Tensor(const BlobTensorPtr& tensor) : tensor_(tensor) {}        // NOLINT
  Tensor(BlobTensorPtr&& tensor) : tensor_(std::move(tensor)) {}  // NOLINT
  static Tensor New(DeviceType device, uint64_t size, DataType dtype = DataType::kFloat) {
    if (device == kCPU) {
      return Tensor(new CpuTensor(dtype, size));
    }
    return Tensor();
  }
  inline const BlobTensor* operator->() const { return tensor_.get(); }
  inline BlobTensor* operator->() { return tensor_.get(); }
  [[nodiscard]] inline const BlobTensorPtr& ptr() const { return tensor_; }
  inline operator const BlobTensorPtr&() const { return tensor_; }  // NOLINT
  inline operator BlobTensorPtr&() { return tensor_; }              // NOLINT
};

}  // namespace fineflow::python_api
#endif  // FINEFLOW_API_PYTHON_PY_TENSOR_H_
