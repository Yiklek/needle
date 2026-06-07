module;
#include <latch>
#include <thread>
export module fineflow.core.executor.cpu_executor;
import std;
import std.compat;
import fineflow.core.executor.cpu_kernel;
import fineflow.core.executor.cpu_range;
import fineflow.core.executor.cpu_actor;

export namespace fineflow {

class CpuExecutor;

class CpuGraphBuilder {
public:
    explicit CpuGraphBuilder(CpuExecutor* exec) : exec_(exec) {}
    CpuGraphBuilder& stage(std::string, CpuKernel& k, CpuRange g, CpuRange l) {
        stages_.push_back({std::ref(k), g, l}); return *this;
    }
    CpuGraphBuilder& barrier() { return *this; }
    void submit();
private:
    CpuExecutor* exec_;
    struct Stage { std::reference_wrapper<CpuKernel> kernel; CpuRange global; CpuRange local; };
    std::vector<Stage> stages_;
};

class CpuStream {
public:
    explicit CpuStream(CpuExecutor* exec) : exec_(exec) {}
    void enqueue(CpuKernel& k, CpuRange g, CpuRange l);
    void wait();
private:
    CpuExecutor* exec_;
};

class CpuExecutor {
public:
    static std::unique_ptr<CpuExecutor> Create(size_t n) {
        return std::unique_ptr<CpuExecutor>(new CpuExecutor(n));
    }
    void enqueue(CpuKernel& kernel, CpuRange global, CpuRange local) {
        size_t nw = global.work_group_count(local);
        std::latch latch(static_cast<ptrdiff_t>(nw));
        for (size_t wg = 0; wg < nw; wg++) {
            pool_.push_back(std::thread([this, &kernel, global, local, wg, &latch] {
                actors_[wg % actors_.size()]
                    ->execute_wg(kernel, (int)wg, global, local);
                latch.count_down();
            }));
        }
        latch.wait();
        for (auto& t : pool_) if (t.joinable()) t.join();
        pool_.clear();
    }
    void wait() {}
    CpuGraphBuilder build_graph() { return CpuGraphBuilder(this); }
    std::unique_ptr<CpuStream> create_stream() {
        return std::unique_ptr<CpuStream>(new CpuStream(this));
    }
private:
    explicit CpuExecutor(size_t n) {
        for (size_t i = 0; i < n; i++)
            actors_.push_back(std::make_unique<CpuActor>(256));
    }
    std::vector<std::thread> pool_;
    std::vector<std::unique_ptr<CpuActor>> actors_;
};

inline void CpuGraphBuilder::submit() {
    for (auto& s : stages_) exec_->enqueue(s.kernel.get(), s.global, s.local);
}
inline void CpuStream::enqueue(CpuKernel& k, CpuRange g, CpuRange l) { exec_->enqueue(k, g, l); }
inline void CpuStream::wait() { exec_->wait(); }

}  // namespace fineflow
