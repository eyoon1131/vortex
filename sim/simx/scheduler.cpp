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

#include "scheduler_impl.h"
#include "scheduler_ml.h"
#include "emulator.h"

using namespace vortex;

namespace {

bool isSchedulable(
    const WarpMask& active_warps,
    const WarpMask& stalled_warps,
    size_t wid) {
    return active_warps.test(wid) && !stalled_warps.test(wid);
}

}

// LinearScanScheduler Implementation
int LinearScanScheduler::selectWarp(
    const WarpMask& active_warps,
    const WarpMask& stalled_warps,
    const std::vector<warp_t>& warps,
    uint64_t cycle) {
    (void)cycle;

    size_t nw = warps.size();

    // Original first-ready policy: scan from warp 0 every time.
    for (size_t wid = 0; wid < nw; ++wid) {
        if (isSchedulable(active_warps, stalled_warps, wid)) {
            return wid;
        }
    }

    return -1;  // No ready warp
}

// RoundRobinScheduler Implementation
int RoundRobinScheduler::selectWarp(
    const WarpMask& active_warps,
    const WarpMask& stalled_warps,
    const std::vector<warp_t>& warps,
    uint64_t cycle) {
    (void)cycle;

    size_t nw = warps.size();
    if (nw == 0) {
        return -1;
    }

    size_t start_warp = last_selected_warp_ < 0
        ? 0
        : (static_cast<size_t>(last_selected_warp_) + 1) % nw;

    for (size_t offset = 0; offset < nw; ++offset) {
        size_t wid = (start_warp + offset) % nw;
        if (isSchedulable(active_warps, stalled_warps, wid)) {
            last_selected_warp_ = static_cast<int>(wid);
            return static_cast<int>(wid);
        }
    }

    return -1;
}

void RoundRobinScheduler::reset() {
    last_selected_warp_ = -1;
}

// GTOScheduler Implementation
GTOScheduler::GTOScheduler(uint32_t num_warps)
    : current_warp_(-1) {
    last_executed_cycle_.resize(num_warps, 0);
}

int GTOScheduler::selectWarp(
    const WarpMask& active_warps,
    const WarpMask& stalled_warps,
    const std::vector<warp_t>& warps,
    uint64_t cycle) {
    (void)cycle;

    size_t nw = warps.size();

    // Check if current warp is still ready
    if (current_warp_ != -1 &&
        (size_t)current_warp_ < nw &&
        active_warps.test(current_warp_) &&
        !stalled_warps.test(current_warp_) &&
        !warps.at(current_warp_).ibuffer.empty()) {
        // Continue with current warp (greedy)
        return current_warp_;
    }

    // Current warp is no longer ready, pick oldest
    int best_warp = -1;
    uint64_t oldest_cycle = UINT64_MAX;

    for (size_t wid = 0; wid < nw; ++wid) {
        bool warp_active = active_warps.test(wid);
        bool warp_stalled = stalled_warps.test(wid);
        if (warp_active && !warp_stalled) {
            if (last_executed_cycle_.at(wid) < oldest_cycle) {
                best_warp = wid;
                oldest_cycle = last_executed_cycle_.at(wid);
            }
        }
    }

    current_warp_ = best_warp;
    return best_warp;
}

void GTOScheduler::notifyExecution(int wid, uint64_t cycle) {
    if (wid >= 0 && (size_t)wid < last_executed_cycle_.size()) {
        last_executed_cycle_.at(wid) = cycle;
    }
}

void GTOScheduler::reset() {
    for (auto& cycle : last_executed_cycle_) {
        cycle = 0;
    }
    current_warp_ = -1;
}

// MLScheduler Implementation
MLScheduler::MLScheduler(uint32_t num_warps)
    : bias_(0.2f)
    , current_warp_(-1) {
    last_executed_cycle_.resize(num_warps, 0);
    stall_count_.resize(num_warps, 0);
    execute_count_.resize(num_warps, 0);

    // Initialize perceptron weights with balanced approach
    // These weights are optimized to work across different workload types
    weights_[FEAT_AGE] = 1.2f;              // Strong fairness - favor warps not recently issued
    weights_[FEAT_IBUFFER_DEPTH] = 0.7f;    // Prefer decoded micro-ops that are ready now
    weights_[FEAT_MEMORY_PRESSURE] = -0.9f; // Penalize recently backpressured warps
    weights_[FEAT_STALL_HISTORY] = -0.4f;   // Mildly penalize chronically stalled warps
}

std::array<float, MLScheduler::NUM_FEATURES> MLScheduler::extractFeatures(
    int warp_id,
    const std::vector<warp_t>& warps,
    uint64_t cycle) const {

    std::array<float, NUM_FEATURES> features = {0, 0, 0, 0};

    if (warp_id < 0 || warp_id >= (int)warps.size()) {
        return features;
    }

    const auto& warp = warps.at(warp_id);

    // Feature 0: Age (cycles since last execution, normalized)
    uint64_t age = cycle - last_executed_cycle_.at(warp_id);
    features[FEAT_AGE] = std::min(100.0f, (float)age) / 100.0f;

    // Feature 1: Instruction buffer depth (normalized to [0, 1])
    int ibuffer_size = warp.ibuffer.size();
    features[FEAT_IBUFFER_DEPTH] = std::min(32.0f, (float)ibuffer_size) / 32.0f;

    // Feature 2: Memory pressure (stall rate, estimated)
    int total_attempts = std::max(1, execute_count_.at(warp_id) + stall_count_.at(warp_id));
    float stall_rate = (float)stall_count_.at(warp_id) / (float)total_attempts;
    features[FEAT_MEMORY_PRESSURE] = stall_rate;

    // Feature 3: Stall history (normalized count of stalls)
    float normalized_stalls = std::min(20.0f, (float)stall_count_.at(warp_id)) / 20.0f;
    features[FEAT_STALL_HISTORY] = normalized_stalls;

    return features;
}

float MLScheduler::scoreWarp(const std::array<float, NUM_FEATURES>& features) const {
    // Perceptron: score = bias + sum(weight_i * feature_i)
    float score = bias_;
    for (int i = 0; i < NUM_FEATURES; i++) {
        score += weights_[i] * features[i];
    }

    // Sigmoid activation to bound score in [0, 1]
    return 1.0f / (1.0f + std::exp(-score));
}

int MLScheduler::selectWarp(
    const WarpMask& active_warps,
    const WarpMask& stalled_warps,
    const std::vector<warp_t>& warps,
    uint64_t cycle) {

    size_t nw = warps.size();

    // Score all schedulable warps. Pick the one with highest score.
    int best_warp = -1;
    float best_score = -1.0f;

    for (size_t wid = 0; wid < nw; ++wid) {
        bool warp_active = active_warps.test(wid);
        bool warp_stalled = stalled_warps.test(wid);

        if (warp_active && !warp_stalled) {
            std::array<float, NUM_FEATURES> features =
                extractFeatures(wid, warps, cycle);
            float score = scoreWarp(features);
            if ((int)wid == current_warp_) {
                score += 0.05f;
            }

            if (score > best_score) {
                best_score = score;
                best_warp = wid;
            }
        }
    }

    current_warp_ = best_warp;
    return best_warp;
}

void MLScheduler::notifyExecution(int wid, uint64_t cycle) {
    if (wid >= 0 && wid < (int)last_executed_cycle_.size()) {
        last_executed_cycle_.at(wid) = cycle;
        execute_count_.at(wid)++;
    }
}

void MLScheduler::notifyStall(int wid, uint64_t cycle) {
    __unused(cycle);
    if (wid >= 0 && wid < (int)stall_count_.size()) {
        stall_count_.at(wid)++;
    }
}

void MLScheduler::reset() {
    std::fill(last_executed_cycle_.begin(), last_executed_cycle_.end(), 0);
    std::fill(stall_count_.begin(), stall_count_.end(), 0);
    std::fill(execute_count_.begin(), execute_count_.end(), 0);
    current_warp_ = -1;
}
