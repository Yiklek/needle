#ifndef FINEFLOW_CORE_BLOB_TENSOR_H_
#define FINEFLOW_CORE_BLOB_TENSOR_H_

#define IS_CPU_NATIVE(t, type_native) (std::is_same_v<t, type_native>)
#define GET_0(a0, a1, ...) a0
#define MAP_IS_CPU_NATIVE(tuple) IS_CPU_NATIVE(T, GET_0 tuple)

#define FF_COMPOSE_READEBLE_TENSOR(class)                                                \
public:                                                                                  \
  [[nodiscard]] const Shape& shape() const override { return tensor_attrs_.shape_; };    \
  [[nodiscard]] const Stride& stride() const override { return tensor_attrs_.stride_; }; \
  [[nodiscard]] DataType dtype() const override { return tensor_attrs_.dtype_; };        \
                                                                                         \
protected:                                                                               \
  TensorAttrsHolder tensor_attrs_;

#define FF_COMPOSE_WRITABLE_TENSOR(class)                                                                   \
  static_assert(&class ::tensor_attrs_ != nullptr, FF_PP_STRINGIZE(class) "must compose readable tensor."); \
                                                                                                            \
public:                                                                                                     \
  Shape& shapeMut() override { return tensor_attrs_.shape_; };                                              \
  Stride& strideMut() override { return tensor_attrs_.stride_; };

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
#endif  // FINEFLOW_CORE_BLOB_TENSOR_H_
