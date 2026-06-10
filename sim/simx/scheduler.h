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

#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include "types.h"

namespace vortex {

struct warp_t;
class Instr;

class WarpScheduler {
public:
    virtual ~WarpScheduler() = default;

    // Select which warp's instruction to execute from ibuffer
    // Returns warp ID (0..N-1) or -1 if no warp ready
    virtual int selectWarp(
        const WarpMask& active_warps,
        const WarpMask& stalled_warps,
        const std::vector<warp_t>& warps,
        uint64_t cycle
    ) = 0;

    // Notify scheduler of instruction execution (for ML models)
    virtual void notifyExecution(int, uint64_t) {}

    // Notify scheduler that a warp experienced pipeline backpressure.
    virtual void notifyStall(int, uint64_t) {}

    // Get scheduler name for logging/debugging
    virtual std::string name() const = 0;

    // Reset scheduler state
    virtual void reset() {}
};

}
