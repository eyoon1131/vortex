#pragma once

#include "scheduler.h"
#include <vector>
#include <array>
#include <cmath>
#include <cstdint>

namespace vortex {

/**
 * @class MLScheduler
 * @brief Machine Learning-based Warp Scheduler
 *
 * Uses a perceptron model to predict best warp based on features:
 * - Age (cycles since last execution)
 * - IBuf Depth (ready instructions)
 * - Memory Pressure (stall rate)
 * - Stall History (count of recent stalls)
 */
class MLScheduler : public WarpScheduler {
public:
    explicit MLScheduler(uint32_t num_warps);

    int selectWarp(
        const WarpMask& active_warps,
        const WarpMask& stalled_warps,
        const std::vector<warp_t>& warps,
        uint64_t cycle
    ) override;

    void notifyExecution(int wid, uint64_t cycle) override;

    void notifyStall(int wid, uint64_t cycle) override;

    std::string name() const override { return "ml"; }

    void reset() override;

private:
    // Feature indices
    static constexpr int NUM_FEATURES = 4;
    static constexpr int FEAT_AGE = 0;
    static constexpr int FEAT_IBUFFER_DEPTH = 1;
    static constexpr int FEAT_MEMORY_PRESSURE = 2;
    static constexpr int FEAT_STALL_HISTORY = 3;

    // Perceptron weights (pre-trained for GEMM-like workloads)
    std::array<float, NUM_FEATURES> weights_;
    float bias_;

    // Per-warp tracking
    std::vector<uint64_t> last_executed_cycle_;
    std::vector<int> stall_count_;
    std::vector<int> execute_count_;
    int current_warp_;

    // Helper methods
    std::array<float, NUM_FEATURES> extractFeatures(
        int warp_id,
        const std::vector<warp_t>& warps,
        uint64_t cycle) const;

    float scoreWarp(const std::array<float, NUM_FEATURES>& features) const;
};

}
