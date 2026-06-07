module;
export module fineflow.core.executor.cpu_range;
import std;
import std.compat;
export namespace fineflow {
struct CpuRange {
    size_t size;
    static CpuRange Global(size_t n) { return {n}; }
    static CpuRange Local(size_t n)  { return {n}; }
    size_t work_group_count(const CpuRange& local) const {
        return size / local.size + (size % local.size ? 1 : 0);
    }
};
}  // namespace fineflow
