#ifndef FINEFLOW_CORE_BLOBTENSOR_H_
#define FINEFLOW_CORE_BLOBTENSOR_H_
#include "fineflow/core/common/data_type.pb.h"
#include "fineflow/core/common/device_type.pb.h"
#include "fineflow/core/common/exception.h"
#include "fineflow/core/common/util.h"
#include "fineflow/core/tensor.h"
namespace fineflow {

inline Shape GetShape(DataType dtype, uint64_t buffer_size) {
  return {static_cast<ssize_t>(buffer_size / **DataTypeSizeRegistryMgr::Get().GetValue(dtype))};
}
inline uint64_t GetBufferSize(DataType dtype, const Shape& shape) {
  return **DataTypeSizeRegistryMgr::Get().GetValue(dtype) * GetElementCount(shape);
}
class ReadableBlobTensorTrait : public ReadableTensorTrait {
  [[nodiscard]] virtual uint64_t bufferSize() const = 0;
  [[nodiscard]] virtual uint64_t offset() const = 0;
  [[nodiscard]] virtual void* rawPtr() const = 0;
  [[nodiscard]] virtual DeviceType device() const = 0;
};
#define FF_COMPOSE_READABLE_BLOB(class)                                            \
public:                                                                            \
  [[nodiscard]] uint64_t bufferSize() const override { return blob_.buffer_size; } \
  [[nodiscard]] uint64_t offset() const override { return blob_.offset; }          \
  [[nodiscard]] void* rawPtr() const override { return blob_.buffer; }             \
  [[nodiscard]] DeviceType device() const override { return blob_.device; };       \
                                                                                   \
protected:                                                                         \
  Blob blob_;

#define FF_COMPOSE_WRITABLE_BLOB(class)                                                           \
  static_assert(&class ::blob_ != nullptr, FF_PP_STRINGIZE(class) "must compose readable blob."); \
  [[nodiscard]] uint64_t& offsetMut() override { return blob_.offset; };                          \
  void*& rawPtrMut() override { return blob_.buffer; };
struct Blob {
  void* buffer = nullptr;
  uint64_t buffer_size = 0;
  uint64_t offset = 0;
  DeviceType device = DeviceType::kInvalidDevice;
};

class ReadableBlobTensor : public virtual ReadableBlobTensorTrait {
  FF_COMPOSE_READEBLE_TENSOR(ReadableBlobTensor)
  FF_COMPOSE_READABLE_BLOB(ReadableBlobTensor)
  FF_DISALLOW_COPY(ReadableBlobTensor);

public:
  ReadableBlobTensor(DeviceType device, DataType dtype, uint64_t buffer_size, uint64_t offset = 0)
      : ReadableBlobTensor(device, dtype, GetShape(dtype, buffer_size)) {
    blob_.offset = offset;
  }
  ReadableBlobTensor(DeviceType device, DataType dtype, const Shape& shape)
      : tensor_attrs_(dtype, shape, GetCompactStride(shape)) {
    blob_.device = device;
    blob_.buffer_size = GetBufferSize(dtype, shape);
    blob_.offset = 0;
  }

  template <typename T = void>
  const T* castPtr() const {
    checkDataType<T>();
    return reinterpret_cast<T*>(rawPtr());
  }
  template <typename T>
  void checkDataType() const {
    if (!static_cast<bool>(std::is_same_v<T, void>) && !static_cast<bool>(std::is_same_v<T, char>) &&
        dtype() != DataType::kChar && dtype() != GetDataType<T>::value) {
      LOG(err) << "tensor data_type mismatched. value: " << DataType_Name(dtype())
               << ", template T:" << DataType_Name(GetDataType<T>::value);
    }
  }

protected:
  ReadableBlobTensor(ReadableBlobTensor&& other) noexcept
      : blob_(std::move(other.blob_)), tensor_attrs_(std::move(other.tensor_attrs_)) {
    other.reset();
  }

  void reset() { blob_.buffer = nullptr; }
};
class WritableBlobTensorTrait : public virtual ReadableBlobTensorTrait, public virtual WritableTensorTrait {
  [[nodiscard]] virtual uint64_t& offsetMut() = 0;
  virtual void*& rawPtrMut() = 0;
};

class WritableBlobTensor : public ReadableBlobTensor, public WritableBlobTensorTrait {
  FF_COMPOSE_WRITABLE_TENSOR(WritableBlobTensor)
  FF_COMPOSE_WRITABLE_BLOB(WritableBlobTensor)
  FF_DISALLOW_COPY(WritableBlobTensor);

public:
  using ReadableBlobTensor::ReadableBlobTensor;

  template <typename T = void>
  T*& castPtrMut() {
    checkDataType<T>();
    return reinterpret_cast<T*&>(rawPtrMut());
  }

protected:
  WritableBlobTensor(WritableBlobTensor&& other) noexcept : ReadableBlobTensor(std::move(other)) {}
};
class LeakedBlobTensor : public WritableBlobTensor {
  explicit LeakedBlobTensor(WritableBlobTensor&& tensor) : WritableBlobTensor(std::move(tensor)) {}
};

template <class T>
class LeakableTensor {
public:
  void leak() && { leak_ = true; };
  [[nodiscard]] bool leaked() const { return leak_; }

private:
  bool leak_ = false;
};
struct RawAllocator {
  static void* allocate(size_t n) {
    void* ptr = nullptr;
    ptr = std::aligned_alloc(64, n);
    if (ptr) return ptr;
    ptr = std::malloc(n);
    if (ptr) return ptr;
    throw RuntimeException(fmt::format("alloc {} bytes failed", n));
  }
  static void deallocate(void* p) { free(p); }
  static void copy(void* dst, void* src, size_t size) { std::memcpy(dst, src, size); }
};

class BlobTensor : public WritableBlobTensor, public LeakableTensor<BlobTensor> {
  FF_DISALLOW_COPY(BlobTensor)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
  // NOTE: Performance will be degraded if the destructor is virtual.
  //       So please do NOT implement custom destructor in any child classes of BlobTensor,
  //       and every fields of child classes should be of POD type.
#pragma GCC diagnostic pop
protected:
  using WritableBlobTensor::WritableBlobTensor;
};

template <class Allocator = RawAllocator, class Enable = std::enable_if_t<std::is_base_of_v<RawAllocator, Allocator>>>
class AllocableBlobTensor : public BlobTensor {
public:
  AllocableBlobTensor(DeviceType device, DataType dtype, uint64_t buffer_size, uint64_t offset = 0)
      : BlobTensor(device, dtype, buffer_size, offset) {
    alloc();
  }
  AllocableBlobTensor(DeviceType device, DataType dtype, const Shape& shape) : BlobTensor(device, dtype, shape) {
    alloc();
  }
  ~AllocableBlobTensor() {
    if (!leaked()) {
      release();
    }
  };

  AllocableBlobTensor(const AllocableBlobTensor& other)
      : AllocableBlobTensor(other.device(), other.dtype(), other.bufferSize(), other.offset()) {
    syncTensorAttrs(other);

    Allocator::copy(rawPtrMut(), other.rawPtr(), other.bufferSize());
  }
  AllocableBlobTensor(AllocableBlobTensor&& other) noexcept : BlobTensor(std::move(other)) {}

  AllocableBlobTensor& operator=(const AllocableBlobTensor& other) {
    if (this == &other) {
      return *this;
    }
    release();
    syncAttrs(other);

    alloc();
    Allocator::copy(rawPtrMut(), other.rawPtr(), other.bufferSize());
  }

  AllocableBlobTensor& operator=(AllocableBlobTensor&& other) noexcept {
    release();
    syncAttrs(other);

    blob_.buffer = other.rawPtr();
    other.reset();
  }

private:
  /* sync* methods only used in copy/move */

  inline void syncTensorAttrs(const AllocableBlobTensor& other) {
    // no sync dtype

    shapeMut() = other.shape();
    strideMut() = other.stride();
  }
  inline void syncBlob(const AllocableBlobTensor& other) {
    // no sync buffer

    blob_.device = other.device();
    blob_.buffer_size = other.bufferSize();
    blob_.offset = other.offset();
  }

  inline void syncAttrs(const AllocableBlobTensor& other) {
    syncTensorAttrs(other);
    syncBlob(other);
  }
  inline void alloc() {
    if (!rawPtr() && bufferSize() > 0) {
      rawPtrMut() = Allocator::allocate(bufferSize());
      LOG(trace) << "allocate: " << rawPtr();
    }
  }
  inline void release() {
    if (rawPtr()) {
      Allocator::deallocate(rawPtr());
      LOG(trace) << "deallocate: " << rawPtr();
      reset();
    }
  }
};

using BlobTensorPtr = std::shared_ptr<fineflow::BlobTensor>;

}  // namespace fineflow
#endif  // FINEFLOW_CORE_BLOBTENSOR_H_
