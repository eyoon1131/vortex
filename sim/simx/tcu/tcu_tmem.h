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

#include "types.h"
#include "tensor_cfg.h"
#include <unordered_set>

#ifdef VX_CFG_TCU_TMEM_ENABLE

namespace vortex {

// Tensor memory: the UMMA accumulator store, plus its column allocator.
//
// TMEM is architecturally per-SM, not per-tensor-core. Living one level down
// inside TcuUnit is correct because Vortex instantiates one TcuUnit per Core;
// if that ever changes, this object should move up to Core.
class TcuTmem {
public:
  static constexpr uint32_t kCols = VX_CFG_TCU_TMEM_COLS;

  using wg_cfg = vortex::tensor::wgmma_config_t<VX_CFG_NUM_THREADS,
                                                vortex::tensor::fp32,
                                                vortex::tensor::fp32>;
  static constexpr uint32_t kLanes = wg_cfg::xtileM * VX_CFG_NUM_TCU_BLOCKS;
  static_assert(kLanes <= 128, "TMEM lanes exceed cap");

  // One physical bank per TCU block
  static constexpr uint32_t kBanks = VX_CFG_NUM_TCU_BLOCKS;

  struct PerfStats {
    uint64_t bank_stalls = 0;    // cycles a requester lost its bank to a conflict
    uint64_t hazard_stalls = 0;  // cycles a read was held by the RAW interlock
  };

  TcuTmem();

  void reset();

  // First-fit free-list column allocator. Multiple warpgroups (different
  // CTAs) can hold disjoint, concurrently-live allocations.
  //
  // CTA-scoped idempotency: every warp of a CTA can call alloc()
  // independently and get the same handle back, so the kernel needs no
  // elected-thread + shared-memory broadcast. The first call for a given
  // cta_id allocates; later calls from sibling warps return the cached handle.
  uint32_t alloc(uint32_t ncols, int32_t cta_id);

  // Mirrors alloc(): the range is only actually freed once every warp of the
  // CTA has called dealloc.
  void dealloc(uint32_t handle, int32_t cta_id, uint32_t wid, uint32_t cta_size);

  // Live-allocation width in columns, or 0 if `handle` is not live.
  uint32_t alloc_ncols(uint32_t handle) const;

  // Bounds-checked element access.
  uint32_t read(uint32_t lane, uint32_t col) const;
  void write(uint32_t lane, uint32_t col, uint32_t value);

  const PerfStats& perf_stats() const { return perf_stats_; }

  // Geometry
  static constexpr uint32_t kBankLanes = wg_cfg::xtileM;  // == TCU_WG_TILE_M
  static constexpr uint32_t kWordCols = wg_cfg::tcN;      // == TCU_TC_N
  static constexpr uint32_t kArbW = 2 * kBanks;           // == ARB_W

  static uint32_t bank_of(uint32_t lane_base) { return lane_base / kBankLanes; }
  static uint32_t word_of(uint32_t col_base) { return col_base / kWordCols; }

  // One requester's bid for one cycle. `valid` false means no request.
  struct BankReq {
    bool valid = false;
    uint32_t lane_base = 0;
    uint32_t col_base = 0;
  };

  // Compute occupies pool indices [0, kBanks) and ldst [kBanks, 2*kBanks).
  struct CycleReqs {
    std::array<BankReq, kBanks> compute_rd{};  // UMMA accumulator read
    std::array<BankReq, kBanks> ldst_rd{};     // TMEM_LD
    std::array<BankReq, kBanks> compute_wr{};  // UMMA accumulator writeback
    std::array<BankReq, kBanks> ldst_wr{};     // TMEM_ST
  };

  // Read grants are the previous arb_step's. Write grants are this arb_step's.
  // An ungranted requester must retry without popping.
  struct Grants {
    std::bitset<kBanks> compute_rd;
    std::bitset<kBanks> ldst_rd;
    std::bitset<kBanks> compute_wr;
    std::bitset<kBanks> ldst_wr;
  };

  // Publish last cycle's registered read grants, then arbitrate this cycle's
  // requests. Stepped once per cycle from TcuUnit::on_tick(), before the
  // per-block consume pass.
  void arb_step(const CycleReqs& reqs);

  const Grants& grants() const { return grants_; }

  // A granted TMEM_ST whose result cannot retire this cycle keeps its request
  // asserted.
  bool won_ldst_wr(uint32_t block) const { return ldst_wr_won_.test(block); }

  // Drop the latch.
  void clear_ldst_wr_win(uint32_t block) { ldst_wr_won_.reset(block); }

  // TODO: ALLOC/DEALLOC serialization against the column allocator
  //       (VX_tcu_tmem_alloc: one winner per cycle).
  // TODO: the wr_track RAW interlock, LANDQ_SIZE deep, keyed on
  //       (lane_base, col_base).

private:
  void validate_lane_col(uint32_t lane, uint32_t col) const;

  std::array<std::array<uint32_t, kCols>, kLanes> data_{};

  // Free-list allocator state: {start_col, ncols} ranges, and handle->ncols
  // for active allocations.
  std::vector<std::pair<uint32_t, uint32_t>> free_{{0, kCols}};
  std::unordered_map<uint32_t, uint32_t> allocs_;

  // CTA-scoped idempotent-alloc bookkeeping.
  std::unordered_map<int32_t, uint32_t> cta_handle_;
  // Distinct wids that have called dealloc for this CTA.
  std::unordered_map<int32_t, std::unordered_set<uint32_t>> cta_dealloc_warps_;

  // Arbitration state. One round-robin arbiter per bank per direction.
  std::vector<Arbiter> rd_arb_;
  std::vector<Arbiter> wr_arb_;

  // Per-bank one-hot read grant, registered one cycle (bank_rd_grant_onehot_d).
  std::array<std::bitset<kArbW>, kBanks> rd_grant_onehot_d_{};

  std::bitset<kBanks> ldst_wr_won_;  // == ldst_wr_won_r
  Grants grants_{};

  PerfStats perf_stats_;
};

} // namespace vortex

#endif // VX_CFG_TCU_TMEM_ENABLE
