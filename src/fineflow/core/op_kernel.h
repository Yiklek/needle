#ifndef FINEFLOW_CORE_OP_KERNEL_H_
#define FINEFLOW_CORE_OP_KERNEL_H_

#include "fineflow/core/blob_tensor.h"
#include "fineflow/core/common/device_type.pb.h"
#include "fineflow/core/common/fmt.hpp"
#include "fineflow/core/common/hash_container.h"
#include "fineflow/core/common/registry_manager.hpp"
#include "fineflow/core/common/result.hpp"
#include "fineflow/core/common/util.h"

namespace fineflow {

class KernelComputeContext {
public:
  [[nodiscard]] std::string opName() const { return std::string(); }
  Ret<std::shared_ptr<BlobTensor>> fetchTensor(const std::string& name, int32_t index) {
    auto key = std::make_pair(name, index);
    auto it = arg2tensor_.find(key);
    CHECK_OR_RETURN(it != arg2tensor_.end()) << "Not found tensor: " << key;
    return it->second;
  }

private:
public:
  HashMap<std::pair<std::string, int32_t>, std::shared_ptr<BlobTensor>> arg2tensor_;
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

#define REGISTER_KERNEL(device, kernel_factory_type) \
  REGISTER_KEY_VALUE(device, std::make_unique<kernel_factory_type>());
}  // namespace fineflow
#endif  // !fineflow_CORE_OP_KERNEL_H_
