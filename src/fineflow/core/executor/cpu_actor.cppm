module;
#include <barrier>
export module fineflow.core.executor.cpu_actor;
import std;
import std.compat;
import fineflow.core.executor.cpu_kernel;
import fineflow.core.executor.cpu_range;
export namespace fineflow {
class CpuActor {
public:
    explicit CpuActor(size_t wg_size) : wg_barrier_(wg_size) {}
    void execute_wg(CpuKernel& kernel, int wg_id, CpuRange global, CpuRange local) {
        size_t start = wg_id * local.size;
        size_t end = std::min(start + local.size, global.size);
        for (size_t i = start; i < end; i++)
            kernel.execute(static_cast<int>(i), wg_id, static_cast<int>(i - start), nullptr);
    }
    std::barrier<>& barrier() { return wg_barrier_; }
private:
    std::barrier<> wg_barrier_;
};
}  // namespace fineflow
