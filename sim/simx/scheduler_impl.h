// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "scheduler.h"
#include <vector>
#include <cstdint>

namespace vortex {

// Linear scan scheduler.
// Always scans from warp 0 and selects the first schedulable warp.
class LinearScanScheduler : public WarpScheduler {
public:
    int selectWarp(
        const WarpMask& active_warps,
        const WarpMask& stalled_warps,
        const std::vector<warp_t>& warps,
        uint64_t cycle
    ) override;

    std::string name() const override { return "LinearScan"; }
};

// True round-robin scheduler.
// Starts each scan after the previously selected warp.
class RoundRobinScheduler : public WarpScheduler {
public:
    int selectWarp(
        const WarpMask& active_warps,
        const WarpMask& stalled_warps,
        const std::vector<warp_t>& warps,
        uint64_t cycle
    ) override;

    std::string name() const override { return "RoundRobin"; }

    void reset() override;

private:
    int last_selected_warp_ = -1;
};

// Greedy Then Oldest scheduler
// Greedily continues current warp if ready, otherwise picks oldest
class GTOScheduler : public WarpScheduler {
public:
    GTOScheduler(uint32_t num_warps);

    int selectWarp(
        const WarpMask& active_warps,
        const WarpMask& stalled_warps,
        const std::vector<warp_t>& warps,
        uint64_t cycle
    ) override;

    void notifyExecution(int wid, uint64_t cycle) override;

    std::string name() const override { return "GTO"; }

    void reset() override;

private:
    std::vector<uint64_t> last_executed_cycle_;
    int current_warp_;
};

} // namespace vortex
