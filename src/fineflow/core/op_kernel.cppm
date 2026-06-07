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

enum class AttrType { kInt, kFloat, kString, kInts, kFloats, kStrings };

using AttrValue = std::variant<
    int64_t, double, std::string,
    std::vector<int64_t>, std::vector<double>, std::vector<std::string>>;

using AttrMap = std::unordered_map<std::string, AttrValue>;

class KernelComputeContext {
public:
  [[nodiscard]] KernelComputeContext(DeviceType device_type, DataType dtype)
      : device_type_(device_type), dtype_(dtype) {}
  [[nodiscard]] std::string opName() const { return std::string(); }
  Ret<BlobTensorView> fetchTensor(const std::string& name, size_t index) {
    auto key = std::make_pair(name, index);
    auto it = arg2tensor_.find(key);
    CHECK_OR_RETURN(it != arg2tensor_.end()) << "Not found tensor: " << key;
    return it->second;
  }
   void insertTensor(const std::string& name, size_t index, const BlobTensorView& tensor) {
    arg2tensor_.insert({{name, index}, tensor});
  }
  void setAttrs(AttrMap attrs) { attrs_ = std::move(attrs); }
  [[nodiscard]] const AttrMap& attrs() const { return attrs_; }
  [[nodiscard]] DeviceType device() const { return device_type_; }
  [[nodiscard]] DataType dtype() const { return dtype_; }

private:
  AttrMap attrs_;
  HashMap<std::pair<std::string, size_t>, BlobTensorView> arg2tensor_;
  DeviceType device_type_;
  DataType dtype_;
};

class OpKernel {
public:
  FF_DISALLOW_COPY_AND_MOVE(OpKernel);
  virtual ~OpKernel() = default;

  virtual void compute(KernelComputeContext& ctx) const { LOG(err) << ctx.opName() << " :UNIMPLEMENTED"; }

protected:
  OpKernel() = default;

private:
  template <typename T, typename... Args>
  friend std::unique_ptr<OpKernel> NewOpKernel(Args&&... args);
};
template <typename T, typename... Args>
std::unique_ptr<OpKernel> NewOpKernel(Args&&... args) {
  return std::unique_ptr<OpKernel>(new T(std::forward<Args>(args)...));
}

template <typename T, typename D>
std::unique_ptr<T> NewKernalFromHandlers(const std::map<D, std::function<std::unique_ptr<T>()>>& handlers,
                                         const D& key) {
  const auto iter = handlers.find(key);
  if (iter != handlers.end()) {
    return iter->second();
  }
  return nullptr;
}

}  // namespace fineflow
