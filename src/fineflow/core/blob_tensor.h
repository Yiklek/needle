#ifndef FINEFLOW_CORE_BLOBTENSOR_H_
#define FINEFLOW_CORE_BLOBTENSOR_H_
#include "fineflow/core/tensor.h"
namespace fineflow {
class BlobTensor : public Tensor {
public:
  [[nodiscard]] virtual uint64_t bufferSize() const = 0;
  [[nodiscard]] virtual uint64_t offset() const = 0;
  [[nodiscard]] virtual uint64_t& offsetMut() = 0;
  ~BlobTensor() override = default;
};

using BlobTensorPtr = std::shared_ptr<fineflow::BlobTensor>;

}  // namespace fineflow
#endif  // FINEFLOW_CORE_BLOBTENSOR_H_
