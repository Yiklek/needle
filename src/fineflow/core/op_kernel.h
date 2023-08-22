#ifndef FINEFLOW_CORE_OP_KERNEL_H_
#define FINEFLOW_CORE_OP_KERNEL_H_
#include <iostream>

#include "fineflow/core/common/hash_container.h"
#include "fineflow/core/common/result.hpp"
#include "fineflow/core/common/util.h"
#include "fineflow/core/tensor.h"
namespace fineflow {

class KernelComputeContext {
public:
  [[nodiscard]] std::string opName() const { return std::string(); }
  Ret<std::shared_ptr<Tensor>> fetchTensor(const std::string& name, int32_t index) {
    auto it = arg2tensor_.find({name, index});
    if (it != arg2tensor_.end()) {
      return it->second;
    }
    return Fail(Error::ValueNotFoundError());
  }

private:
public:
  HashMap<std::pair<std::string, int32_t>, std::shared_ptr<Tensor>> arg2tensor_;
};

class OpKernel {
public:
  FF_DISALLOW_COPY_AND_MOVE(OpKernel);
  virtual ~OpKernel() = default;

  // virtual std::shared_ptr<OpKernelState> CreateOpKernelState(KernelInitContext* ctx) const {
  //   return std::shared_ptr<OpKernelState>();
  // }
  //
  // virtual std::shared_ptr<OpKernelCache> InitOpKernelCache(KernelCacheContext* ctx) const {
  //   return std::shared_ptr<OpKernelCache>();
  // }

  // virtual void InitOpKernelCacheWithFlags(KernelCacheContext* ctx, int8_t flag,
  //                                         std::shared_ptr<OpKernelCache>* cache_ptr) const {
  //   *cache_ptr = InitOpKernelCache(ctx);
  // }

  // virtual void Compute(KernelComputeContext* ctx, OpKernelState*, const OpKernelCache*) const { Compute(ctx); }
  virtual void compute(KernelComputeContext* ctx) const { std::cout << ctx->opName() << " :UNIMPLEMENTED"; }
  // virtual void InferShape(KernelInferContext* ctx) const;
  // virtual bool AlwaysComputeWhenAllOutputsEmpty() const = 0;
  // virtual bool IsKernelLaunchSynchronized() const { return true; }

  // bool has_state_or_cache() const { return has_state_or_cache_; }

protected:
  // OpKernel() : has_state_or_cache_(true) {}
  OpKernel() = default;

private:
  template <typename T, typename... Args>
  friend OpKernel* NewOpKernel(Args&&... args);
  // bool has_state_or_cache_;
};
template <typename T, typename... Args>
OpKernel* NewOpKernel(Args&&... args) {
  OpKernel* ptr = new T(std::forward<Args>(args)...);
  // ptr->has_state_or_cache_ = !(std::is_same<decltype(&OpKernel::CreateOpKernelState),
  //                                           decltype(&T::CreateOpKernelState)>::value
  //                              && std::is_same<decltype(&OpKernel::InitOpKernelCache),
  //                                              decltype(&T::InitOpKernelCache)>::value
  //                              && std::is_same<decltype(&OpKernel::InitOpKernelCacheWithFlags),
  //                                              decltype(&T::InitOpKernelCacheWithFlags)>::value);
  return ptr;
}
}  // namespace fineflow
#endif  // !fineflow_CORE_OP_KERNEL_H_
