#ifndef FINEFLOW_API_PYTHON_FINEFLOW_H_
#define FINEFLOW_API_PYTHON_FINEFLOW_H_
#include "fineflow/core/common/data_type.h"
#include "fineflow/core/common/preprocess.h"
#include "fineflow/core/common/registry.h"
#define TYPE_NUMPY_FORMAT(type) py::format_descriptor<type>::format()
#define TYPE_NUMPY_TUPLE(type, dtype) (dtype, TYPE_NUMPY_FORMAT(type))

#define REGISTER_NUMPY_FORMAT(type_proto, format)             \
  REGISTER_KEY_VALUE_T(DataTypeToFormat, type_proto, format); \
  REGISTER_KEY_VALUE_T(FormatToDataType, format, type_proto);

#define MAP_REGISTER_NUMPY_FORMAT(tuple) REGISTER_NUMPY_FORMAT FF_PP_FORWARD(TYPE_NUMPY_TUPLE, FF_TUPLE_TO_ENUM(tuple))

#define REGISTER_FILL_PYFUNCTOR(type_cpp, type_proto) \
  m.def(func_name, std::function(PyFunctor<void, Tensor&, type_cpp>(func_name)));
#define MAP_REGISTER_FILL_PYFUNCTOR(tuple) REGISTER_FILL_PYFUNCTOR tuple

#define REGISTER_ASSIGN_PYFUNCTOR(type_cpp, type_proto) \
  m.def(func_name, std::function(PyFunctor<Tensor, Tensor&, type_cpp>(func_name)));
#define MAP_REGISTER_ASSIGN_PYFUNCTOR(tuple) REGISTER_ASSIGN_PYFUNCTOR tuple

#endif  // FINEFLOW_API_PYTHON_FINEFLOW_H_
