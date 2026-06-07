#include "gtest/gtest.h"
import std;
import fineflow.core.executor.cpu_range;
import fineflow.core.executor.cpu_kernel;
import fineflow.core.executor.cpu_actor;
import fineflow.core.executor.cpu_executor;

using namespace fineflow;

// --- CpuRange ---
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

// --- CpuKernel ---
TEST(CpuKernel, FromLambdaSingleWorkItem) {
    float result = 0;
    auto kernel = CpuKernel::FromLambda(
        [&](int gid, int, int) { result = gid * 2.0f; });
    void* args[] = {&result};
    kernel.execute(3, 0, 0, args);
    EXPECT_FLOAT_EQ(result, 6.0f);
}
TEST(CpuKernel, FromLambdaMultipleWorkItems) {
    std::vector<float> r(4, 0);
    auto kernel = CpuKernel::FromLambda(
        [&](int gid, int, int) { r[gid] = gid * 1.0f; });
    for (int i = 0; i < 4; i++) {
        void* args[] = {r.data()};
        kernel.execute(i, 0, i % 2, args);
    }
    for (int i = 0; i < 4; i++) EXPECT_FLOAT_EQ(r[i], i * 1.0f);
}

// --- CpuActor ---
TEST(CpuActor, ExecuteWorkGroup) {
    std::atomic<int> counter{0};
    auto kernel = CpuKernel::FromLambda(
        [&](int, int, int) { counter.fetch_add(1, std::memory_order_relaxed); });
    CpuActor actor(64);
    actor.execute_wg(kernel, 0, CpuRange::Global(128), CpuRange::Local(64));
    EXPECT_EQ(counter.load(), 64);
}

// --- CpuExecutor ---
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
    EXPECT_EQ(sum.load(), 4950);
}

TEST(CpuExecutor, StreamsConcurrent) {
    std::atomic<int> a{0}, b{0};
    auto ka = CpuKernel::FromLambda([&](int, int, int) { a.fetch_add(1); });
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
