#ifndef FINEFLOW_CORE_TENSOR_H_
#define FINEFLOW_CORE_TENSOR_H_
#include <iostream>
// #include <cstdlib>
#include "fineflow/core/common/data_type.h"
#include "fineflow/core/common/log.h"
namespace fineflow {

using Shape = std::vector<ssize_t>;
using Stride = std::vector<ssize_t>;
inline int64_t GetElementCount(const Shape& shape) {
  return std::accumulate(shape.begin(), shape.end(), 1L, std::multiplies<>());
}
inline Stride GetCompactStride(const Shape& shape) {
  ssize_t stride = 1;
  Stride ret(shape.size(), 0);
  for (int i = shape.size() - 1; i >= 0; --i) {
    ret[i] = stride;
    stride *= shape[i];
  }
  return ret;
}
class Tensor {
public:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
  // NOTE: Performance will be degraded if the destructor is virtual.
  //       So please do NOT implement custom destructor in any child classes of user_op::Tensor,
  //       and every fields of child classes should be of POD type.
  virtual ~Tensor() = default;
#pragma GCC diagnostic pop

  [[nodiscard]] virtual const Shape& shape() const = 0;
  virtual Shape& shapeMut() = 0;
  [[nodiscard]] virtual const Stride& stride() const = 0;
  virtual Stride& strideMut() = 0;
  [[nodiscard]] virtual DataType dtype() const = 0;
  // virtual MemoryFormat memory_format() const = 0;
  // virtual const MemoryCase& mem_case() const = 0;
  [[nodiscard]] virtual const void* rawPtr() const = 0;
  virtual void* rawPtrMut() = 0;

  [[nodiscard]] int64_t elementCount() const { return GetElementCount(shape()); }
  template <typename T = void>
  const T* castPtr() const {
    checkDataType<T>();
    return reinterpret_cast<const T*>(rawPtr());
  }

  template <typename T = void>
  T* castPtrMut() {
    checkDataType<T>();
    return reinterpret_cast<T*>(rawPtrMut());
  }

protected:
  template <typename T>
  void checkDataType() const {
    if (!static_cast<bool>(std::is_same_v<T, void>) && !static_cast<bool>(std::is_same_v<T, char>) &&
        dtype() != DataType::kChar && dtype() != GetDataType<T>::value) {
      LOG(err) << "tensor data_type mismatched. value: " << DataType_Name(dtype())
               << ", template T:" << DataType_Name(GetDataType<T>::value);
    }
  }
};
}  // namespace fineflow

#endif
