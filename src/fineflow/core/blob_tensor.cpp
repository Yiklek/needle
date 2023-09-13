#include "fineflow/core/blob_tensor.h"
#include "fineflow/core/cpu/cpu_tensor.h"

namespace fineflow {

BlobTensorView::BlobTensorView(const BlobTensorPtr& other) : WritableBlobTensor(*other.get()) { tensor_ = other; }

}  // namespace fineflow
