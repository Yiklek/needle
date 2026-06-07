module;
#include <cstdint>
export module fineflow.core.executor.cpu_kernel;
import std;
import std.compat;
export namespace fineflow {
class CpuKernel {
public:
    template<typename F>
    static CpuKernel FromLambda(F&& fn) {
        return CpuKernel(
            [fn = std::forward<F>(fn)](int gid, int wgid, int wid, void** args) {
                fn(gid, wgid, wid);
            });
    }
    void execute(int global_id, int wg_id, int wi_id, void** args) {
        fn_(global_id, wg_id, wi_id, args);
    }
private:
    std::function<void(int, int, int, void**)> fn_;
    explicit CpuKernel(std::function<void(int, int, int, void**)> fn) : fn_(std::move(fn)) {}
};
}  // namespace fineflow
