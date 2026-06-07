module;
#include <latch>
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>
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
        auto sched = pool_.get_scheduler();
        auto snd = stdexec::schedule(sched)
                 | stdexec::bulk(stdexec::par, nw,
                     [this, &kernel, global, local](size_t wg_id) {
                         actors_[wg_id % actors_.size()]
                             ->execute_wg(kernel, (int)wg_id, global, local);
                     });
        stdexec::sync_wait(std::move(snd));
    }
    void wait() {}
    CpuGraphBuilder build_graph() { return CpuGraphBuilder(this); }
    std::unique_ptr<CpuStream> create_stream() {
        return std::unique_ptr<CpuStream>(new CpuStream(this));
    }
private:
    explicit CpuExecutor(size_t n) : pool_(n) {
        for (size_t i = 0; i < n; i++)
            actors_.push_back(std::make_unique<CpuActor>(256));
    }
    exec::static_thread_pool pool_;
    std::vector<std::unique_ptr<CpuActor>> actors_;
};

inline void CpuGraphBuilder::submit() {
    for (auto& s : stages_) exec_->enqueue(s.kernel.get(), s.global, s.local);
}
inline void CpuStream::enqueue(CpuKernel& k, CpuRange g, CpuRange l) { exec_->enqueue(k, g, l); }
inline void CpuStream::wait() { exec_->wait(); }

}  // namespace fineflow
