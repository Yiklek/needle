#ifndef FINEFLOW_CORE_TENSOR_H_
#define FINEFLOW_CORE_TENSOR_H_
#include <iostream>
#include <utility>
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
  if (shape.empty()) {
    return {};
  }
  ssize_t stride = 1;
  Stride ret(shape.size(), 0);
  for (int i = shape.size() - 1; i >= 0; --i) {
    ret[i] = stride;
    stride *= shape[i];
  }
  return ret;
}
class ReadableTensorTrait {
public:
  [[nodiscard]] virtual const Shape& shape() const = 0;
  [[nodiscard]] virtual const Stride& stride() const = 0;
  [[nodiscard]] virtual DataType dtype() const = 0;
  [[nodiscard]] int64_t elementCount() const { return GetElementCount(shape()); }
  [[nodiscard]] bool isScalar() const { return elementCount() == 1 && shape().empty() && stride().empty(); }
};
struct WritableTensorTrait {
public:
  virtual Shape& shapeMut() = 0;    // { return shape_; };
  virtual Stride& strideMut() = 0;  //  { return stride_; };
};

class TensorAttrsHolder {
public:
  TensorAttrsHolder(const DataType& dtype, const Shape& shape)
      : TensorAttrsHolder(dtype, shape, GetCompactStride(shape)) {}
  TensorAttrsHolder(const DataType& dtype, Shape shape, Stride stride)
      : dtype_(dtype), shape_(std::move(shape)), stride_(std::move(stride)) {}

  DataType dtype_;
  Shape shape_;
  Stride stride_;
};

#define FF_COMPOSE_READEBLE_TENSOR(class)                                                \
public:                                                                                  \
  [[nodiscard]] const Shape& shape() const override { return tensor_attrs_.shape_; };    \
  [[nodiscard]] const Stride& stride() const override { return tensor_attrs_.stride_; }; \
  [[nodiscard]] DataType dtype() const override { return tensor_attrs_.dtype_; };        \
                                                                                         \
protected:                                                                               \
  TensorAttrsHolder tensor_attrs_;

#define FF_COMPOSE_WRITABLE_TENSOR(class)                                                                   \
  static_assert(&class ::tensor_attrs_ != nullptr, FF_PP_STRINGIZE(class) "must compose readable tensor."); \
                                                                                                            \
public:                                                                                                     \
  Shape& shapeMut() override { return tensor_attrs_.shape_; };                                              \
  Stride& strideMut() override { return tensor_attrs_.stride_; };

}  // namespace fineflow
#endif
