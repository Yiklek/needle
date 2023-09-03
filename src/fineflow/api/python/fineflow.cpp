#include "fineflow/api/python/py_functor.hpp"
#include "fineflow/api/python/py_tensor.h"
#include "fineflow/core/common/device_type.pb.h"
#include "fineflow/core/common/error_util.h"
#include "fineflow/core/common/registry_manager.hpp"
#include "fineflow/core/common/result.hpp"
#include "fineflow/core/common/types_tuple.h"
#include "fineflow/core/cpu/cpu_tensor.h"
#include "fineflow/core/functional.h"
#include "fmt/ranges.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
namespace fineflow::python_api {
namespace py = pybind11;
struct DataTypeToFormat;
struct FormatToDataType;

auto ToNumpy(Tensor &a) {
  auto numpy_strides = a->stride();
  auto elem_size = *DataTypeSizeRegistryMgr::Get().GetValue(a->dtype()).value();
  const auto &format = **RegistryMgr<DataType, std::string, DataTypeToFormat>::Get().GetValue(a->dtype());
  std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                 [elem_size](auto &c) { return c * elem_size; });
  return py::array(
      py::buffer_info(reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(a->rawPtrMut()) + a->offset() * elem_size),
                      elem_size, format, a->shape().size(), a->shape(), numpy_strides));
}

auto FromNumpy(const py::array &a, DeviceType devide_type) {
  auto b = a.request();
  const auto &dtype = **RegistryMgr<std::string, DataType, FormatToDataType>::Get().GetValue(b.format);
  auto out = Tensor::New(devide_type, b.size * b.itemsize, dtype);
  std::memcpy(out->rawPtrMut(), b.ptr, out->bufferSize());
  auto elem_size = *RegistryMgr<DataType, size_t>::Get().GetValue(dtype).value();
  auto numpy_strides = b.strides;
  std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                 [elem_size](auto &c) { return c / elem_size; });
  out->shapeMut() = b.shape;
  out->strideMut() = numpy_strides;
  return out;
}

PYBIND11_MODULE(PYBIND11_CURRENT_MODULE_NAME, m) {
  auto ewise_add = std::function(PyFunctor<Tensor, const Tensor &, const Tensor &>("add"));
  m.attr("__device_name__") = "cpu_fine";
  // m.attr("__tile_size__") = TILE;

  // m.def("ewise_add", EwiseAdd);
  m.def("ewise_add", ewise_add);

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
      .def("ptr", [](Tensor &self) { return reinterpret_cast<size_t>(self->castPtr()); })
      .def_property_readonly("size", [](Tensor &self) { return self->bufferSize(); })
      .def("to_numpy", ToNumpy);

  m.def("to_numpy", ToNumpy);
  m.def("from_numpy", FromNumpy, py::arg("array"), py::arg("devide_type") = DeviceType::kCPU);
};
namespace {

#define TYPE_NUMPY_FORMAT(type) py::format_descriptor<type>::format()
#define TYPE_NUMPY_TUPLE(type, dtype) (dtype, TYPE_NUMPY_FORMAT(type))

#define REGISTRE_NUMPY_FORMAT(type_proto, format)             \
  REGISTER_KEY_VALUE_T(DataTypeToFormat, type_proto, format); \
  REGISTER_KEY_VALUE_T(FormatToDataType, format, type_proto);

#define REGISTRE_NUMPY_FORMAT_TUPLE(tuple, i)                             \
  FF_PP_FORWARD(REGISTRE_NUMPY_FORMAT, BOOST_PP_TUPLE_ENUM(FF_PP_FORWARD( \
                                           TYPE_NUMPY_TUPLE, BOOST_PP_TUPLE_ENUM(BOOST_PP_TUPLE_ELEM(i, tuple)))))

// for i in [0, CPU_PRIMITIVE_NATIVE_TYPE_TUPLE)
#define BOOST_PP_LOCAL_LIMITS (0, BOOST_PP_TUPLE_SIZE(CPU_PRIMITIVE_NATIVE_TYPE_TUPLE) - 1)
#define BOOST_PP_LOCAL_MACRO(i) REGISTRE_NUMPY_FORMAT_TUPLE(CPU_PRIMITIVE_NATIVE_TYPE_TUPLE, i)
#include BOOST_PP_LOCAL_ITERATE()

}  // namespace
}  // namespace fineflow::python_api
