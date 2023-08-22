#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fineflow/api/python/py_tensor.h"
#include "fineflow/core/common/auto_register.hpp"
#include "fineflow/core/common/device_type.pb.h"
#include "fineflow/core/cpu/cpu_tensor.h"
#include "fineflow/core/kernels/add_kernel.h"
namespace fineflow {
PYBIND11_MODULE(FineflowPyLib, m) {
  namespace py = pybind11;

  m.def("add", [](float a, float b) {
    auto ta = std::make_shared<CpuTensor>(DataType::kFloat, sizeof(a));
    auto tb = std::make_shared<CpuTensor>(DataType::kFloat, sizeof(a));
    auto tc = std::make_shared<CpuTensor>(DataType::kFloat, sizeof(a));
    *ta->castPtrMut<decltype(a)>() = a;
    *tb->castPtrMut<decltype(b)>() = b;
    KernelComputeContext ctx;
    ctx.arg2tensor_.insert({{"in", 0}, ta});
    ctx.arg2tensor_.insert({{"in", 1}, tb});
    ctx.arg2tensor_.insert({{"out", 0}, tc});
    AddKernelFactory* f = NewObj<DeviceType, AddKernelFactory>(DeviceType::kCPU);
    f->create(DataType::kFloat)->compute(&ctx);
    auto r = *tc->castPtr<float>();
    return r;
  });
  py::class_<python_api::Tensor>(m, "Tensor").def(py::init([]() { return python_api::Tensor(); }));
};

}  // namespace fineflow
