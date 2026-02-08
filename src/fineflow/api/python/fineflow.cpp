
#include "fineflow/api/python/fineflow.h"

#include "fineflow/core/common/data_type.h"
#include "fineflow/core/common/map.h"
#include "fineflow/core/common/result.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

import fineflow.api.python.py_functor;
import fineflow.api.python.py_tensor;
import fineflow.core.common.exception;
import fineflow.core.common.fmt;
import fineflow.core.common.data_type_proto;
import fineflow.core.common.device_type_proto;
import fineflow.core.common.data_type;
import fineflow.core.common.registry_manager;
import fineflow.core.common.error;

namespace fineflow::python_api {
namespace py = pybind11;
struct DataTypeToFormat;
struct FormatToDataType;

inline const std::string& GetTypeFormat(DataType dtype) {
  TRY_ASSIGN_CATCH(auto r, FF_PP_ALL(RegistryMgr<DataType, std::string, DataTypeToFormat>::Get().GetValue(dtype)),
                   { ThrowError(e); });
  return *r;
}
inline size_t GetTypeElemSize(DataType dtype) {
  TRY_ASSIGN_CATCH(auto r, FF_PP_ALL(DataTypeSizeRegistryMgr::Get().GetValue(dtype)), { ThrowError(e); });
  return *r;
}

inline DataType GetFormatType(const std::string& format) {
  auto f = format;
  // https://github.com/pybind/pybind11/issues/1908
  if constexpr (sizeof(void*) == 8) {  // 64bit
    if (f == "l") {
      f = "q";
    }
  }
  TRY_ASSIGN_CATCH(auto r, FF_PP_ALL(RegistryMgr<std::string, DataType, FormatToDataType>::Get().GetValue(f)),
                   { ThrowError(e); });
  return *r;
}

inline auto ToNumpy(Tensor& a) {
  auto elem_size = **DataTypeSizeRegistryMgr::Get().GetValue(a->dtype());
  const auto& format = GetTypeFormat(a->dtype());
  auto& t = *a;
  if (a->offset() > 0) {
    auto compact = PyFunctor<Tensor, const Tensor&>("compact");
    *t = **compact(a);
  }

  auto numpy_strides = t->stride();
  std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                 [elem_size](auto& c) { return c * elem_size; });  // numpy sitrde is by bytes.

  if (t->offset() > 0) {
    throw RuntimeException("numpy offset must be 0.");
  }
  return py::array(
      py::buffer_info(reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(t->rawPtrMut()) + t->offset() * elem_size),
                      elem_size, format, t->shape().size(), t->shape(), numpy_strides));
}

inline auto FromNumpy(const py::array& a, DeviceType device_type) {
  auto b = a.request();
  const auto dtype = GetFormatType(b.format);
  auto out = Tensor::New(device_type, b.size * b.itemsize, dtype);
  std::memcpy(out->rawPtrMut(), b.ptr, out->bufferSize());
  auto elem_size = GetTypeElemSize(dtype);
  auto numpy_strides = b.strides;
  std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                 [elem_size](auto& c) { return c / elem_size; });  // numpy sitrde is by bytes.
  out->shapeMut() = b.shape;
  out->strideMut() = numpy_strides;
  return out;
}
}  // namespace fineflow::python_api
namespace fineflow::python_api {
namespace py = pybind11;
void RegisterFill(py::module_& m) {
  const auto* func_name = "fill";
  auto fill = std::function(PyFunctor<void, Tensor&, const Tensor&>(func_name));
  m.def(func_name, fill);
  // must capture fill function as value
  m.def(func_name, [=](Tensor& t, const py::array& a) { return fill(t, FromNumpy(a, t->device())); });

  MAP(MAP_REGISTER_FILL_PYFUNCTOR, CPU_PRIMITIVE_NATIVE_TYPE_ENUM)
}

void RegisterAssign(py::module_& m) {
  const auto* func_name = "assign";
  auto assign = std::function(PyFunctor<void, Tensor&, const Tensor&>(func_name));
  m.def(func_name, assign);
  // must capture fill function as value
  m.def(func_name, [=](Tensor& t, const py::array& a) { return assign(t, FromNumpy(a, t->device())); });

  MAP(MAP_REGISTER_ASSIGN_PYFUNCTOR, CPU_PRIMITIVE_NATIVE_TYPE_ENUM)
  // #include "assign_def.gen"
}

void RegisterAdd(py::module_& m) {
  auto ewise_add = std::function(PyFunctor<Tensor, const Tensor&, const Tensor&>("add"));
  m.def("ewise_add", ewise_add);
}

PYBIND11_MODULE(PYBIND11_CURRENT_MODULE_NAME, m) {
  py::register_exception_translator([](std::exception_ptr p) {  // NOLINT
    try {
      if (p) std::rethrow_exception(p);
    } catch (const TypeException& e) {
      throw py::type_error(e.what());
    } catch (const IndexException& e) {
      throw py::index_error(e.what());
    } catch (const NotImplementedException& e) {
      PyErr_SetString(PyExc_NotImplementedError, e.what());
    }
  });
  m.attr("__device_name__") = "cpu_fine";
  // m.attr("__tile_size__") = TILE;
  RegisterFill(m);
  RegisterAdd(m);
  RegisterAssign(m);
  py::enum_<DeviceType>(m, "DeviceType")
      .value("cpu", DeviceType::kCPU)
      .value("cuda", DeviceType::kCUDA)
      .value("none", DeviceType::kInvalidDevice)
      .value("mock", DeviceType::kMockDevice);

  py::class_<Tensor>(m, "Tensor")
      .def(py::init([](uint64_t size) { return Tensor::New(DeviceType::kCPU, size); }),
           py::return_value_policy::take_ownership)
      .def(py::init([](uint64_t size, DataType dtype) { return Tensor::New(DeviceType::kCPU, size, dtype); }),
           py::return_value_policy::take_ownership)
      .def("ptr", [](Tensor& self) { return reinterpret_cast<size_t>(self->castPtr()); })
      .def_property_readonly("size", [](Tensor& self) { return self->bufferSize(); })
      .def("to_numpy", ToNumpy);

  m.def("to_numpy", ToNumpy);
  m.def("from_numpy", FromNumpy, py::arg("array"), py::arg("device_type") = DeviceType::kCPU);
};
namespace {
MAP(MAP_REGISTER_NUMPY_FORMAT, CPU_PRIMITIVE_NATIVE_TYPE_ENUM)
}  // namespace
}  // namespace fineflow::python_api
