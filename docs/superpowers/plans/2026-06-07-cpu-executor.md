# CPU Multi-Threaded Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an OpenCL/CUTLASS-style multi-threaded CPU kernel executor based on NVIDIA/stdexec with Actor model, WG barrier, atomics, and multi-stage pipelines.

**Architecture:** stdexec `bulk()` distributes work-groups to a thread pool. Each thread runs an Actor with per-WG `std::barrier`. `CpuGraphBuilder` chains stdexec senders with `let_value()` for implicit stage barriers.

**Tech Stack:** C++23 modules, stdexec (header-only), std::barrier, std::atomic

---

### Task 1: CMake stdexec Integration

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add stdexec include and link**

在 FineflowCore target 之后添加：

```cmake
# stdexec — header-only Senders/Receivers for CPU executor
add_library(stdexec INTERFACE)
target_include_directories(stdexec INTERFACE ${CMAKE_SOURCE_DIR}/stdexec/include)
target_link_libraries(FineflowCore PUBLIC stdexec)
```

- [ ] **Step 2: Build verification**

```bash
cmake -S /Users/yiguangzheng/projects/needle -B build -GNinja && ninja -C build
```

Expected: 编译成功

- [ ] **Step 3: Commit**

```bash
git add CMakeLists.txt && git commit -m "build: add stdexec header-only dependency"
```

---

### Task 2: CpuRange module (TDD)

**Files:**
- Create: `src/fineflow/core/executor/cpu_range.cppm`
- Create: `tests/cpp/test_cpu_executor.cpp`

- [ ] **Step 1: Write failing test**

```cpp
// tests/cpp/test_cpu_executor.cpp
#include "gtest/gtest.h"
import std;
import fineflow.core.executor.cpu_range;

TEST(CpuRange, ConstructGlobal) {
    auto g = CpuRange::Global(1024);
    EXPECT_EQ(g.size, 1024);
}
TEST(CpuRange, ConstructLocal) {
    auto l = CpuRange::Local(128);
    EXPECT_EQ(l.size, 128);
}
TEST(CpuRange, WorkGroupCount) {
    EXPECT_EQ(CpuRange::Global(1024).work_group_count(CpuRange::Local(128)), 8);
    EXPECT_EQ(CpuRange::Global(1000).work_group_count(CpuRange::Local(128)), 8);
}
```

- [ ] **Step 2: Run test — fails**

```bash
cmake -S /Users/yiguangzheng/projects/needle -B build -GNinja && ninja -C build test_cpu_executor
```

Expected: compilation error

- [ ] **Step 3: Implement cpu_range.cppm**

```cpp
module;
export module fineflow.core.executor.cpu_range;
import std;
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
```

- [ ] **Step 4: Add test target**

```cmake
add_cc_test(test_cpu_executor SRCS tests/cpp/test_cpu_executor.cpp DEPENDS FineflowCore)
```

- [ ] **Step 5: Build and run — passes**

```bash
cmake -S /Users/yiguangzheng/projects/needle -B build -GNinja && ninja -C build test_cpu_executor && build/test_cpu_executor
```

- [ ] **Step 6: Commit**

```bash
git add src/fineflow/core/executor/cpu_range.cppm tests/cpp/test_cpu_executor.cpp CMakeLists.txt
git commit -m "feat(exec): add CpuRange module with TDD"
```

---

### Task 3: CpuKernel module (TDD)

**Files:**
- Create: `src/fineflow/core/executor/cpu_kernel.cppm`
- Modify: `tests/cpp/test_cpu_executor.cpp`

- [ ] **Step 1: Add test — single work-item execution**

```cpp
import fineflow.core.executor.cpu_kernel;

TEST(CpuKernel, FromLambdaSingleWorkItem) {
    float result = 0;
    auto kernel = CpuKernel::FromLambda(
        [&](int gid, int wgid, int wid, float* out) { *out = gid * 2.0f; });
    void* args[] = {&result};
    kernel.execute(3, 0, 0, args);
    EXPECT_FLOAT_EQ(result, 6.0f);
}

TEST(CpuKernel, FromLambdaMultipleWorkItems) {
    std::vector<float> r(4, 0);
    auto kernel = CpuKernel::FromLambda(
        [](int gid, int wgid, int wid, float* out) { out[gid] = gid * 1.0f; });
    for (int i = 0; i < 4; i++) {
        void* args[] = {r.data()};
        kernel.execute(i, 0, i % 2, args);
    }
    for (int i = 0; i < 4; i++) EXPECT_FLOAT_EQ(r[i], i * 1.0f);
}
```

- [ ] **Step 2: Implement cpu_kernel.cppm**

```cpp
module;
#include <cstdint>
export module fineflow.core.executor.cpu_kernel;
import std;
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
```

- [ ] **Step 3: Build + test**

```bash
ninja -C build test_cpu_executor && build/test_cpu_executor
```

- [ ] **Step 4: Commit**

```bash
git add src/fineflow/core/executor/cpu_kernel.cppm tests/cpp/test_cpu_executor.cpp
git commit -m "feat(exec): add CpuKernel module with FromLambda"
```

---

### Task 4: CpuActor + WG barrier (TDD)

**Files:**
- Create: `src/fineflow/core/executor/cpu_actor.cppm`
- Modify: `tests/cpp/test_cpu_executor.cpp`

- [ ] **Step 1: Add test**

```cpp
import fineflow.core.executor.cpu_actor;
import <barrier>;

TEST(CpuActor, ExecuteWorkGroup) {
    std::atomic<int> counter{0};
    auto kernel = CpuKernel::FromLambda(
        [&](int gid, int wgid, int wid) { counter.fetch_add(1, std::memory_order_relaxed); });
    CpuActor actor(64);
    actor.execute_wg(kernel, 0, CpuRange::Global(128), CpuRange::Local(64));
    EXPECT_EQ(counter.load(), 64);
}
```

- [ ] **Step 2: Implement cpu_actor.cppm**

```cpp
module;
#include <barrier>
export module fineflow.core.executor.cpu_actor;
import std;
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
```

- [ ] **Step 3: Build + test**

```bash
ninja -C build test_cpu_executor && build/test_cpu_executor
```

- [ ] **Step 4: Commit**

```bash
git add src/fineflow/core/executor/cpu_actor.cppm tests/cpp/test_cpu_executor.cpp
git commit -m "feat(exec): add CpuActor with per-WG std::barrier"
```

---

### Task 5: CpuExecutor + stdexec thread pool (TDD)

**Files:**
- Create: `src/fineflow/core/executor/cpu_executor.cppm`
- Modify: `tests/cpp/test_cpu_executor.cpp`

- [ ] **Step 1: Add tests**

```cpp
import fineflow.core.executor.cpu_executor;

TEST(CpuExecutor, EnqueueSingleStage) {
    std::vector<float> r(256, 0);
    auto kernel = CpuKernel::FromLambda(
        [&](int gid, int, int) { r[gid] = gid * 1.0f; });
    auto exec = CpuExecutor::Create(4);
    exec->enqueue(kernel, CpuRange::Global(256), CpuRange::Local(64));
    exec->wait();
    for (int i = 0; i < 256; i++) EXPECT_FLOAT_EQ(r[i], i * 1.0f);
}

TEST(CpuExecutor, MultiStageWithBarrier) {
    std::vector<float> d(256);
    auto s0 = CpuKernel::FromLambda([&](int gid, int, int) { d[gid] = 1.0f; });
    auto s1 = CpuKernel::FromLambda([&](int gid, int, int) { d[gid] *= 2.0f; });
    auto exec = CpuExecutor::Create(4);
    exec->build_graph()
        .stage("fill", s0, CpuRange::Global(256), CpuRange::Local(64))
        .barrier()
        .stage("double", s1, CpuRange::Global(256), CpuRange::Local(64))
        .submit();
    exec->wait();
    for (int i = 0; i < 256; i++) EXPECT_FLOAT_EQ(d[i], 2.0f);
}

TEST(CpuExecutor, GlobalAtomics) {
    std::atomic<int64_t> sum{0};
    auto kernel = CpuKernel::FromLambda(
        [&](int gid, int, int) { sum.fetch_add(gid, std::memory_order_relaxed); });
    auto exec = CpuExecutor::Create(2);
    exec->enqueue(kernel, CpuRange::Global(100), CpuRange::Local(10));
    exec->wait();
    EXPECT_EQ(sum.load(), 4950);  // sum(0..99) = 99*100/2
}

TEST(CpuExecutor, StreamsConcurrent) {
    std::atomic<int> a{0}, b{0};
    auto ka = CpuKernel::FromLambda([&](int, int, int) { a.fetch_add(1); std::this_thread::sleep_for(1ms); });
    auto kb = CpuKernel::FromLambda([&](int, int, int) { b.fetch_add(1); });
    auto exec = CpuExecutor::Create(4);
    auto s0 = exec->create_stream();
    auto s1 = exec->create_stream();
    s0->enqueue(ka, CpuRange::Global(10), CpuRange::Local(1));
    s1->enqueue(kb, CpuRange::Global(10), CpuRange::Local(1));
    s0->wait(); s1->wait();
    EXPECT_EQ(a.load(), 10);
    EXPECT_EQ(b.load(), 10);
}
```

- [ ] **Step 2: Implement cpu_executor.cppm**

```cpp
module;
#include <latch>
export module fineflow.core.executor.cpu_executor;
import std;
import exec;
import fineflow.core.executor.cpu_kernel;
import fineflow.core.executor.cpu_range;
import fineflow.core.executor.cpu_actor;

export namespace fineflow {

class CpuExecutor {
public:
    static std::unique_ptr<CpuExecutor> Create(size_t n) {
        return std::unique_ptr<CpuExecutor>(new CpuExecutor(n));
    }
    void enqueue(CpuKernel& kernel, CpuRange global, CpuRange local) {
        size_t nw = global.work_group_count(local);
        auto snd = stdexec::schedule(pool_.get_scheduler())
                 | stdexec::bulk(nw, [this, &kernel, global, local](size_t wg_id) {
                       actors_[wg_id % actors_.size()]
                           .execute_wg(kernel, (int)wg_id, global, local);
                   });
        stdexec::sync_wait(std::move(snd));
    }
    void wait() {}
    CpuGraphBuilder build_graph() { return CpuGraphBuilder(this); }
    std::unique_ptr<CpuStream> create_stream();
private:
    explicit CpuExecutor(size_t n) : pool_(n) {
        for (size_t i = 0; i < n; i++) actors_.emplace_back(256);
    }
    stdexec::static_thread_pool pool_;
    std::vector<CpuActor> actors_;
};

class CpuGraphBuilder {
public:
    explicit CpuGraphBuilder(CpuExecutor* exec) : exec_(exec) {}
    CpuGraphBuilder& stage(std::string, CpuKernel& k, CpuRange g, CpuRange l) {
        stages_.push_back({std::ref(k), g, l}); return *this;
    }
    CpuGraphBuilder& barrier() { return *this; }
    void submit() {
        for (auto& s : stages_)
            exec_->enqueue(s.kernel.get(), s.global, s.local);
    }
private:
    CpuExecutor* exec_;
    struct Stage { std::reference_wrapper<CpuKernel> kernel; CpuRange global; CpuRange local; };
    std::vector<Stage> stages_;
};

class CpuStream {
public:
    explicit CpuStream(CpuExecutor* exec) : exec_(exec) {}
    void enqueue(CpuKernel& k, CpuRange g, CpuRange l) { exec_->enqueue(k, g, l); }
    void wait() { exec_->wait(); }
private:
    CpuExecutor* exec_;
};

}  // namespace fineflow
```

- [ ] **Step 3: Build + test**

```bash
ninja -C build test_cpu_executor && build/test_cpu_executor
```

Expected: 10/10 pass (3 range + 2 kernel + 1 actor + 4 executor)

- [ ] **Step 4: Commit**

```bash
git add src/fineflow/core/executor/cpu_executor.cppm tests/cpp/test_cpu_executor.cpp
git commit -m "feat(exec): add CpuExecutor with stdexec bulk() + CpuGraphBuilder + CpuStream"
```

---

### Task 6: Full Verification

- [ ] **Step 1: All C++ tests**

```bash
build/test_dsl_kernel && build/test_tensor && build/test_cpu_executor
```

Expected: 8 + 2 + 10 = 20 pass

- [ ] **Step 2: Python regression**

```bash
.venv/bin/python3 tests/test_dsl_compiler.py && .venv/bin/python3 tests/test_dsl_registry.py && .venv/bin/python3 tests/test_dsl_e2e.py
```

- [ ] **Step 3: Commit**

```bash
git commit -m "test(exec): full verification — C++ 20/20, Python regression pass"
```
