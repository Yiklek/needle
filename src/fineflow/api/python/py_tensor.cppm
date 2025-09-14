module;
#include "fineflow/core/common/log.h"
export module fineflow.api.python.py_tensor;
import std;
// import std.compat;

import fineflow.core.blob_tensor;
import fineflow.core.common.device_type_proto;
import fineflow.core.common.data_type_proto;
import fineflow.core.common.log;
import fineflow.core.common.fmt;
export namespace fineflow::python_api {

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
  static Tensor New(DeviceType device, unsigned long size, DataType dtype = DataType::kFloat) {
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
    LOG(trace) << std::format("Deconstruct python_api tensor. Core tensor (Use count): {} ({}). Buffer: {}",
                              std::ptr(tensor_->ptr().get()), tensor_->ptr().use_count(),
                              std::ptr(tensor_->ptr()->rawPtr()));
    // delete tensor_;
  }
};

}  // namespace fineflow::python_api
