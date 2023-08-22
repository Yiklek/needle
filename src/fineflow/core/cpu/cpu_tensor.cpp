#include "fineflow/core/cpu/cpu_tensor.h"

namespace fineflow {
CpuTensor &fineflow::CpuTensor::operator=(const CpuTensor &other) {
  if (&other == this) {
    return *this;
  }
  release();
  buffer_size_ = other.buffer_size_;
  alloc();
  std::memcpy(buffer_, other.buffer_, buffer_size_);
  return *this;
};

CpuTensor &CpuTensor::operator=(CpuTensor &&other) noexcept {
  if (&other == this) {
    return *this;
  }
  release();
  buffer_size_ = other.buffer_size_;
  buffer_ = other.buffer_;
  other.buffer_ = nullptr;
  return *this;
};
}  // namespace fineflow
