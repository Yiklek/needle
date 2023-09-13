#ifndef FINEFLOW_CORE_BLOBTENSOR_H_
#define FINEFLOW_CORE_BLOBTENSOR_H_
#include "fineflow/core/common/data_type.pb.h"
#include "fineflow/core/common/device_type.pb.h"
#include "fineflow/core/common/exception.h"
#include "fineflow/core/common/util.h"
#include "fineflow/core/tensor.h"
namespace fineflow {

inline Shape GetShape(DataType dtype, uint64_t buffer_size) {
  auto elem_size = **DataTypeSizeRegistryMgr::Get().GetValue(dtype);
  auto count = buffer_size / elem_size;
  if (count <= 1) {
    return {};
  }
  return {static_cast<ssize_t>(count)};
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
  // FF_DISALLOW_COPY(ReadableBlobTensor);

protected:
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

public:
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
  FF_DEFAULT_COPY(ReadableBlobTensor)
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

protected:
  using ReadableBlobTensor::ReadableBlobTensor;

public:
  template <typename T = void>
  T*& castPtrMut() {
    checkDataType<T>();
    return reinterpret_cast<T*&>(rawPtrMut());
  }

protected:
  FF_DEFAULT_COPY_AND_MOVE(WritableBlobTensor);
  // WritableBlobTensor(WritableBlobTensor&& other) noexcept : ReadableBlobTensor(std::move(other)) {}

  /* sync* methods only used in copy/move */

  inline void syncTensorAttrs(const ReadableBlobTensor& other) {
    // no sync dtype

    shapeMut() = other.shape();
    strideMut() = other.stride();
  }
  inline void syncBlob(const ReadableBlobTensor& other) {
    // no sync buffer

    blob_.device = other.device();
    blob_.buffer_size = other.bufferSize();
    blob_.offset = other.offset();
  }

  inline void syncAttrs(const ReadableBlobTensor& other) {
    syncTensorAttrs(other);
    syncBlob(other);
  }
  WritableBlobTensor& operator=(const WritableBlobTensor& other) {
    if (this == &other) return *this;

    syncAttrs(other);
    blob_.buffer = other.rawPtr();

    return *this;
  }
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
class BlobTensor;
using BlobTensorPtr = std::shared_ptr<BlobTensor>;

class BlobTensorView : public WritableBlobTensor {
  BlobTensorPtr tensor_;

public:
  using WritableBlobTensor::operator=;
  FF_DEFAULT_COPY(BlobTensorView)
  BlobTensorView(const BlobTensorPtr& other);  // NOLINT
  [[nodiscard]] const BlobTensorPtr& ptr() const { return tensor_; }
  [[nodiscard]] BlobTensorPtr& ptr() { return tensor_; }
  const BlobTensorPtr& operator*() const { return tensor_; }
  BlobTensorPtr& operator*() { return tensor_; }
};

class BlobTensor : public WritableBlobTensor,
                   public LeakableTensor<BlobTensor>,
                   public std::enable_shared_from_this<BlobTensor> {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
  // NOTE: Performance will be degraded if the destructor is virtual.
  //       So please do NOT implement custom destructor in any child classes of BlobTensor,
  //       and every fields of child classes should be of POD type.
#pragma GCC diagnostic pop
public:
  Ret<BlobTensorPtr> shared() {
    auto ptr = weak_from_this().lock();
    CHECK_OR_RETURN(ptr) << "BlobTensor " << this << " is not shared.";
    return ptr;
  }

  operator BlobTensorView() {  // NOLINT
    return shared_from_this();
  }
  BlobTensorView view() { return *this; }

protected:
  FF_DEFAULT_COPY_AND_MOVE(BlobTensor)
  using WritableBlobTensor::WritableBlobTensor;
};

template <class Allocator = RawAllocator, class Enable = std::enable_if_t<std::is_base_of_v<RawAllocator, Allocator>>>
class AllocableBlobTensor : public BlobTensor {
protected:
  AllocableBlobTensor(DeviceType device, DataType dtype, uint64_t buffer_size, uint64_t offset = 0)
      : BlobTensor(device, dtype, buffer_size, offset) {
    alloc();
  }
  AllocableBlobTensor(DeviceType device, DataType dtype, const Shape& shape) : BlobTensor(device, dtype, shape) {
    alloc();
  }

  // scalar
  AllocableBlobTensor(DeviceType device, DataType dtype) : BlobTensor(device, dtype, {}) { alloc(); }

  template <class T>
  AllocableBlobTensor(DeviceType device, T&& t) : BlobTensor(device, GetDataType<T>::value, {}) {
    alloc();
    *castPtrMut<T>() = t;
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
  inline void alloc() {
    if (!rawPtr() && bufferSize() > 0) {
      rawPtrMut() = Allocator::allocate(bufferSize());
      LOG(trace) << "allocate: " << rawPtr();
    }
  }
  inline void release() {
    if (rawPtr()) {
      Allocator::deallocate(rawPtr());
      LOG(trace) << "deallocate: " << rawPtr() << ". Use count: " << weak_from_this().use_count();
      reset();
    }
  }
};

using BlobTensorPtr = std::shared_ptr<fineflow::BlobTensor>;

}  // namespace fineflow
#endif  // FINEFLOW_CORE_BLOBTENSOR_H_
