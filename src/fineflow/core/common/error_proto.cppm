module;
#include "fineflow/core/common/error.pb.h"
export module fineflow.core.common.error_proto;
import std;
export namespace fineflow {
using ErrorProto = ErrorProto;

// only private use in error. do not use this in other file.
using Descriptor = google::protobuf::Descriptor;
using OneofDescriptor = google::protobuf::OneofDescriptor;
using Reflection = google::protobuf::Reflection;
using FieldDescriptor = google::protobuf::FieldDescriptor;
}  // namespace fineflow
