#ifndef FINEFLOW_API_PYTHON_PY_TENSOR_H_
#define FINEFLOW_API_PYTHON_PY_TENSOR_H_
#include <memory>

#include "fineflow/core/tensor.h"
namespace fineflow::python_api {

class Tensor final {
private:
  std::shared_ptr<fineflow::Tensor> tensor_;
};

}  // namespace fineflow::python_api
#endif  // FINEFLOW_API_PYTHON_PY_TENSOR_H_
