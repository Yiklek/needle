#ifndef FINEFLOW_CORE_CPU_CPU_TENSOR_H_
#define FINEFLOW_CORE_CPU_CPU_TENSOR_H_
#include <cstdlib>

#include "fineflow/core/tensor.h"
#include "fmt/format.h"

namespace fineflow {

class CpuTensor final : public Tensor {
public:
  CpuTensor(DataType dtype, void *buffer, uint64_t buffer_size)
      : dtype_(dtype), buffer_(buffer), buffer_size_(buffer_size) {}

  explicit CpuTensor(DataType dtype, uint64_t buffer_size) : dtype_(dtype), buffer_size_(buffer_size) { alloc(); }

  CpuTensor(const CpuTensor &other) : CpuTensor(other.dtype(), other.buffer_size_) {
    std::memcpy(buffer_, other.buffer_, buffer_size_);
  };
  CpuTensor(CpuTensor &&other) noexcept : CpuTensor(other.dtype(), other.buffer_, other.buffer_size_) {
    other.buffer_ = nullptr;
  }
  CpuTensor &operator=(const CpuTensor &other);
  CpuTensor &operator=(CpuTensor &&) noexcept;
  ~CpuTensor() { release(); }
  [[nodiscard]] const Shape &shape() const override { return shape_; };
  Shape &shapeMut() override { return shape_; };
  [[nodiscard]] const Stride &stride() const override { return stride_; };
  Stride &strideMut() override { return stride_; };
  [[nodiscard]] DataType dtype() const override { return dtype_; };
  [[nodiscard]] const void *rawPtr() const override { return buffer_; };
  void *rawPtrMut() override { return buffer_; };

  [[nodiscard]] uint64_t bufferSize() const { return buffer_size_; }

private:
  void release() {
    if (buffer_) {
      free(buffer_);
    }
  }
  void alloc() {
    if (!buffer_ && buffer_size_ > 0) {
      buffer_ = std::aligned_alloc(64, buffer_size_);
      if (buffer_) return;
      buffer_ = std::malloc(buffer_size_);
      if (buffer_) return;
      std::cerr << fmt::format("alloc {} bytes failed", buffer_size_);
    }
  }
  void *buffer_ = nullptr;
  uint64_t buffer_size_;
  Shape shape_;
  Stride stride_;
  DataType dtype_;
};
}  // namespace fineflow
#endif  // FINEFLOW_CORE_CPU_CPU_TENSOR_H_
