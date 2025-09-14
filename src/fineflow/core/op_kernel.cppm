module;

#include "fineflow/core/common/log.h"
#include "fineflow/core/common/result.h"
#include "fineflow/core/common/util.h"

export module fineflow.core.op_kernel;

import fineflow.core.blob_tensor;
import fineflow.core.common.device_type_proto;
import fineflow.core.common.data_type_proto;
import fineflow.core.common.registry_manager;
import fineflow.core.common.error;
import fineflow.core.common.hash;
import fineflow.core.common.log;
import std;
import std.compat;

export namespace fineflow {

class KernelComputeContext {
public:
  [[nodiscard]] KernelComputeContext(DeviceType device_type, DataType dtype)
      : device_type_(device_type), dtype_(dtype) {}
  std::string opName() const { return std::string(); }
  Ret<BlobTensorView> fetchTensor(const std::string& name, size_t index) {
    auto key = std::make_pair(name, index);
    auto it = arg2tensor_.find(key);
    CHECK_OR_RETURN(it != arg2tensor_.end()) << "Not found tensor: " << key;
    return it->second;
  }
  inline void insertTensor(const std::string& name, size_t index, const BlobTensorView& tensor) {
    arg2tensor_.insert({{name, index}, tensor});
  }
  DeviceType device() const { return device_type_; }
  DataType dtype() const { return dtype_; }

private:
  HashMap<std::pair<std::string, size_t>, BlobTensorView> arg2tensor_;
  DeviceType device_type_;
  DataType dtype_;
};

class OpKernel {
public:
  FF_DISALLOW_COPY_AND_MOVE(OpKernel);
  virtual ~OpKernel() = default;

  virtual void compute(KernelComputeContext* ctx) const { LOG(err) << ctx->opName() << " :UNIMPLEMENTED"; }

protected:
  OpKernel() = default;

private:
  template <typename T, typename... Args>
  friend OpKernel* NewOpKernel(Args&&... args);
};
template <typename T, typename... Args>
OpKernel* NewOpKernel(Args&&... args) {
  OpKernel* ptr = new T(std::forward<Args>(args)...);
  return ptr;
}
template <class T>
using KernelRegistryMgr = RegistryMgr<DeviceType, std::unique_ptr<T>>;

// REGISTER_KEY_WITH_CLASS(DeviceType, std::unique_ptr<OpKernel>, device) \
//     .setValue(std::make_unique<kernel_factory_type>())
#define REGISTER_KERNEL(device, kernel_factory_type) \
  REGISTER_KEY_VALUE(device, std::make_unique<kernel_factory_type>());
}  // namespace fineflow
