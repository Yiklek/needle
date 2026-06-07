module;
export module fineflow.core.executor.parallel_dispatch;
import std;
import std.compat;
import fineflow.core.executor.cpu_kernel;
import fineflow.core.executor.cpu_range;
import fineflow.core.executor.cpu_executor;

export namespace fineflow {

// Singleton executor shared across all CPU kernels
inline CpuExecutor& GetCpuExecutor() {
    static auto exec = CpuExecutor::Create(
        std::thread::hardware_concurrency());
    return *exec;
}

// Dispatch a work-item lambda across global work-items using the shared executor.
// Each work-item processes a single element at index [global_id].
template<typename F>
inline void ParallelDispatch(size_t total_elements, size_t local_size, F&& fn) {
    if (total_elements == 0) return;
    auto kernel = CpuKernel::FromLambda(
        [fn = std::forward<F>(fn)](int gid, int, int) { fn(gid); });
    GetCpuExecutor().enqueue(kernel,
        CpuRange::Global(total_elements), CpuRange::Local(local_size));
}

}  // namespace fineflow
