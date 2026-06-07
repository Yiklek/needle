module;

export module fineflow.core.op_prototype;

import std;
import std.compat;
import fineflow.core.op_kernel;
import fineflow.core.common.registry_manager;

export namespace fineflow {

struct PrototypeTag {};

struct OpParam {
  std::string name;
  enum class IoType { kInput, kOutput };
  IoType io_type;
  int64_t min_count = 1;
  int64_t max_count = 1;
  bool optional = false;
};

struct AttrDef {
  AttrType type;
  AttrValue default_value;
};

struct OpPrototype {
  std::string name;
  std::string domain = "";
  int64_t since_version = 1;
  std::vector<OpParam> inputs;
  std::vector<OpParam> outputs;
  std::unordered_map<std::string, AttrDef> attrs;
};

using OpPrototypeRegistry = RegistryMgr<std::string, OpPrototype, PrototypeTag>;

}  // namespace fineflow
