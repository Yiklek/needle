#include "fineflow/api/python/py_functor.hpp"
#include "fineflow/api/python/py_tensor.h"
#include "fineflow/api/python/tensor_util.h"
#include "fineflow/core/common/exception.h"
#include "fineflow/core/common/fmt.hpp"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace fineflow::python_api {
namespace py = pybind11;
void RegisterFill(py::module_ &m) {
  auto fill = std::function(PyFunctor<void, Tensor &, const Tensor &>("fill"));
  m.def("fill", fill);
  // must capture fill function as value
  m.def("fill", [=](Tensor &t, const py::array &a) { return fill(t, FromNumpy(a, t->device())); });

#define REGISTER_FILL_PYFUNCTOR(type_cpp, type_proto) \
  m.def("fill", std::function(PyFunctor<Tensor, Tensor &, type_cpp>("fill")));
#define FOR_REGISTER_FILL_PYFUNCTOR(i, data, elem) REGISTER_FILL_PYFUNCTOR(elem, _)
#define REGISTER_FILL_TYPE_SEQ (double)(bool)(int64_t)(std::complex<float>)(std::complex<double>)
  BOOST_PP_SEQ_FOR_EACH(FOR_REGISTER_FILL_PYFUNCTOR, _, REGISTER_FILL_TYPE_SEQ)
#undef FOR_REGISTER_FILL_FUNCTOR
#undef REGISTER_FILL_FUNCTOR
}

void RegisterAssign(py::module_ &m) {
  auto assign = std::function(PyFunctor<void, Tensor &, const Tensor &>("assign"));
  const auto *func_name = "assign";
  m.def(func_name, assign);
  // must capture fill function as value
  m.def(func_name, [=](Tensor &t, const py::array &a) { return assign(t, FromNumpy(a, t->device())); });

#define REGISTER_ASSIGN_PYFUNCTOR(type_cpp, type_proto) \
  m.def(func_name, std::function(PyFunctor<Tensor, Tensor &, type_cpp>(func_name)));
#define FOR_REGISTER_ASSIGN_PYFUNCTOR(i, data, elem) REGISTER_ASSIGN_PYFUNCTOR(elem, _)
#define REGISTER_ASSIGN_TYPE_SEQ (double)(bool)(int64_t)(std::complex<float>)(std::complex<double>)
  BOOST_PP_SEQ_FOR_EACH(FOR_REGISTER_ASSIGN_PYFUNCTOR, _, REGISTER_ASSIGN_TYPE_SEQ)
#undef FOR_REGISTER_ASSIGN_PYFUNCTOR
#undef REGISTER_ASSIGN_FUNCTOR
}

void RegisterAdd(py::module_ &m) {
  auto ewise_add = std::function(PyFunctor<Tensor, const Tensor &, const Tensor &>("add"));
  m.def("ewise_add", ewise_add);
}

PYBIND11_MODULE(PYBIND11_CURRENT_MODULE_NAME, m) {
  py::register_exception_translator([](std::exception_ptr p) {  // NOLINT
    try {
      if (p) std::rethrow_exception(p);
    } catch (const TypeException &e) {
      throw py::type_error(e.what());
    } catch (const IndexException &e) {
      throw py::index_error(e.what());
    } catch (const NotImplementedException &e) {
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
      .def("ptr", [](Tensor &self) { return reinterpret_cast<size_t>(self->castPtr()); })
      .def_property_readonly("size", [](Tensor &self) { return self->bufferSize(); })
      .def("to_numpy", ToNumpy);

  m.def("to_numpy", ToNumpy);
  m.def("from_numpy", FromNumpy, py::arg("array"), py::arg("device_type") = DeviceType::kCPU);
};
namespace {

#define TYPE_NUMPY_FORMAT(type) py::format_descriptor<type>::format()
#define TYPE_NUMPY_TUPLE(type, dtype) (dtype, TYPE_NUMPY_FORMAT(type))

#define REGISTER_NUMPY_FORMAT(type_proto, format)             \
  REGISTER_KEY_VALUE_T(DataTypeToFormat, type_proto, format); \
  REGISTER_KEY_VALUE_T(FormatToDataType, format, type_proto);

#define REGISTRE_NUMPY_FORMAT_TUPLE(tuple, i)                             \
  FF_PP_FORWARD(REGISTRE_NUMPY_FORMAT, BOOST_PP_TUPLE_ENUM(FF_PP_FORWARD( \
                                           TYPE_NUMPY_TUPLE, BOOST_PP_TUPLE_ENUM(BOOST_PP_TUPLE_ELEM(i, tuple)))))

#define FOR_REGISTER_NUMPY_FORMAT(i, _, elem) \
  FF_PP_FORWARD(REGISTER_NUMPY_FORMAT, BOOST_PP_TUPLE_ENUM(FF_PP_FORWARD(TYPE_NUMPY_TUPLE, BOOST_PP_TUPLE_ENUM(elem))))
BOOST_PP_SEQ_FOR_EACH(FOR_REGISTER_NUMPY_FORMAT, _, CPU_PRIMITIVE_NATIVE_TYPE_SEQ)
#undef FOR_REGISTER_NUMPY_FORMAT
}  // namespace
}  // namespace fineflow::python_api
