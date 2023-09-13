#ifndef FINEFLOW_API_PYTHON_TENSOR_UTIL_H_
#define FINEFLOW_API_PYTHON_TENSOR_UTIL_H_
#include <string>

#include "fineflow/api/python/py_functor.hpp"
#include "fineflow/api/python/py_tensor.h"
#include "fineflow/core/common/data_type.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
namespace fineflow::python_api {
namespace py = pybind11;
struct DataTypeToFormat;
struct FormatToDataType;

inline const std::string &GetTypeFormat(DataType dtype) {
  TRY_ASSIGN_CATCH(auto r, FF_PP_ALL(RegistryMgr<DataType, std::string, DataTypeToFormat>::Get().GetValue(dtype)),
                   { ThrowError(e); });
  return *r;
}
inline size_t GetTypeElemSize(DataType dtype) {
  TRY_ASSIGN_CATCH(auto r, FF_PP_ALL(RegistryMgr<DataType, size_t>::Get().GetValue(dtype)), { ThrowError(e); });
  return *r;
}

inline DataType GetFormatType(const std::string &format) {
  auto f = format;
  // https://github.com/pybind/pybind11/issues/1908
  if constexpr (sizeof(void *) == 8) {  // 64bit
    if (f == "l") {
      f = "q";
    }
  }
  TRY_ASSIGN_CATCH(auto r, FF_PP_ALL(RegistryMgr<std::string, DataType, FormatToDataType>::Get().GetValue(f)),
                   { ThrowError(e); });
  return *r;
}

inline auto ToNumpy(Tensor &a) {
  auto elem_size = **DataTypeSizeRegistryMgr::Get().GetValue(a->dtype());
  const auto &format = GetTypeFormat(a->dtype());
  auto &t = *a;
  if (a->offset() > 0) {
    auto compact = PyFunctor<Tensor, const Tensor &>("compact");
    *t = **compact(a);
  }

  auto numpy_strides = t->stride();
  std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                 [elem_size](auto &c) { return c * elem_size; });  // numpy sitrde is by bytes.

  if (t->offset() > 0) {
    throw RuntimeException("numpy offset must be 0.");
  }
  return py::array(
      py::buffer_info(reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(t->rawPtrMut()) + t->offset() * elem_size),
                      elem_size, format, t->shape().size(), t->shape(), numpy_strides));
}

inline auto FromNumpy(const py::array &a, DeviceType device_type) {
  auto b = a.request();
  const auto dtype = GetFormatType(b.format);
  auto out = Tensor::New(device_type, b.size * b.itemsize, dtype);
  std::memcpy(out->rawPtrMut(), b.ptr, out->bufferSize());
  auto elem_size = GetTypeElemSize(dtype);
  auto numpy_strides = b.strides;
  std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                 [elem_size](auto &c) { return c / elem_size; });  // numpy sitrde is by bytes.
  out->shapeMut() = b.shape;
  out->strideMut() = numpy_strides;
  return out;
}

}  // namespace fineflow::python_api
#endif  // FINEFLOW_API_PYTHON_TENSOR_UTIL_H_
