#ifndef FINEFLOW_API_PYTHON_PY_TENSOR_H_
#define FINEFLOW_API_PYTHON_PY_TENSOR_H_
#include <memory>

#include "fineflow/core/blob_tensor.h"
#include "fineflow/core/common/device_type.pb.h"
#include "fineflow/core/cpu/cpu_tensor.h"
namespace fineflow::python_api {

class Tensor final {
  using ViewPtr = std::unique_ptr<BlobTensorView>;

private:
  ViewPtr tensor_;

public:
  Tensor(const Tensor& tensor) : tensor_(ViewPtr(new BlobTensorView(**tensor))) {}              // NOLINT
  Tensor(const BlobTensorView& tensor) : tensor_(ViewPtr(new BlobTensorView(tensor))) {}        // NOLINT
  Tensor(BlobTensorView&& tensor) : tensor_(ViewPtr(new BlobTensorView(std::move(tensor)))) {}  // NOLINT
  Tensor(const BlobTensorPtr& tensor) : Tensor(tensor->view()) {}                               // NOLINT
  Tensor& operator=(const Tensor& tensor) {
    if (this == &tensor) {
      return *this;
    }
    *tensor_ = **tensor;
    return *this;
  }
  static Tensor New(DeviceType device, uint64_t size, DataType dtype = DataType::kFloat) {
    if (device == kCPU) {
      return Tensor(CpuTensor::New(dtype, size));
    }
    return Tensor(nullptr);
  }
  const ViewPtr& operator->() const { return tensor_; }
  ViewPtr& operator->() { return tensor_; }

  const ViewPtr& operator*() const { return tensor_; }
  ViewPtr& operator*() { return tensor_; }

  // [[nodiscard]] inline const BlobTensorPtr& ptr() const { return tensor_; }
  inline operator const ViewPtr&() const { return tensor_; }          // NOLINT
  inline operator ViewPtr&() { return tensor_; }                      // NOLINT
  inline operator BlobTensorView() { return *tensor_; }               // NOLINT
  inline operator BlobTensorView&() { return *tensor_; }              // NOLINT
  inline operator const BlobTensorView&() const { return *tensor_; }  // NOLINT
  ~Tensor() {
    LOG(trace) << fmt::format("Deconstruct python_api tensor. Core tensor (Use count): {} ({}). Buffer: {}",
                              fmt::ptr(tensor_->ptr().get()), tensor_->ptr().use_count(),
                              fmt::ptr(tensor_->ptr()->rawPtr()));
    // delete tensor_;
  }
};

}  // namespace fineflow::python_api
#endif  // FINEFLOW_API_PYTHON_PY_TENSOR_H_
