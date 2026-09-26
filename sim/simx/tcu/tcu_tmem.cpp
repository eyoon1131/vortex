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

#include "tcu_tmem.h"

#ifdef VX_CFG_TCU_TMEM_ENABLE

#include "debug.h"
#include <algorithm>
#include <iostream>

using namespace vortex;

namespace {

// VX_rr_arbiter resets reqs_mask to all-ones, so its first grant is the
// lowest set request. SimX's RoundRobinArbiter resets last_grant_ to 0, which
// starts the scan at index 1 and would skip requester 0 on the first
// contended cycle. Prime it with a grant of the top index so both models
// enter their rotation at the same point.
void prime_rr(Arbiter& arb, uint32_t size) {
  BitVector<> top(size);
  top.set(size - 1);
  arb.grant(top);
}

} // namespace

TcuTmem::TcuTmem() {
  rd_arb_.reserve(kBanks);
  wr_arb_.reserve(kBanks);
  for (uint32_t r = 0; r < kBanks; ++r) {
    rd_arb_.emplace_back(ArbiterType::RoundRobin, kArbW);
    wr_arb_.emplace_back(ArbiterType::RoundRobin, kArbW);
  }
  this->reset();
}

void TcuTmem::reset() {
  for (auto& row : data_) row.fill(0);
  free_.assign(1, {0, kCols});
  allocs_.clear();
  cta_handle_.clear();
  cta_dealloc_warps_.clear();
  for (uint32_t r = 0; r < kBanks; ++r) {
    rd_arb_.at(r).reset();
    wr_arb_.at(r).reset();
    prime_rr(rd_arb_.at(r), kArbW);
    prime_rr(wr_arb_.at(r), kArbW);
    rd_grant_onehot_d_.at(r).reset();
  }
  ldst_wr_won_.reset();
  grants_ = Grants{};
  perf_stats_ = PerfStats{};
}

///////////////////////////////////////////////////////////////////////////////

uint32_t TcuTmem::alloc(uint32_t ncols, int32_t cta_id) {
  auto cta_it = cta_handle_.find(cta_id);
  if (cta_it != cta_handle_.end()) {
    uint32_t handle = cta_it->second;
    if (allocs_.at(handle) != ncols) {
      std::cout << "Error: TMEM_ALLOC ncols mismatch for cta_id=" << cta_id
                << " (existing=" << allocs_.at(handle) << ", requested=" << ncols
                << ") — one CTA can only hold one live allocation in this PoC" << std::endl;
      std::abort();
    }
    return handle;
  }

  for (auto it = free_.begin(); it != free_.end(); ++it) {
    if (it->second < ncols) continue;
    uint32_t handle = it->first;
    if (it->second == ncols) {
      free_.erase(it);
    } else {
      it->first  += ncols;
      it->second -= ncols;
    }
    allocs_[handle] = ncols;
    for (uint32_t c = handle; c < handle + ncols; ++c)
      for (auto& row : data_)
        row[c] = 0;
    cta_handle_[cta_id] = handle;
    return handle;
  }
  std::cout << "Error: TMEM allocation failed (ncols=" << ncols
            << ", no free range large enough)" << std::endl;
  std::abort();
}

void TcuTmem::dealloc(uint32_t handle, int32_t cta_id, uint32_t wid, uint32_t cta_size) {
  auto it = allocs_.find(handle);
  if (it == allocs_.end()) {
    std::cout << "Error: TMEM_DEALLOC unknown handle " << handle << std::endl;
    std::abort();
  }
  auto& dealloc_warps = cta_dealloc_warps_[cta_id];
  dealloc_warps.insert(wid);
  if (dealloc_warps.size() < cta_size) {
    return; // other warps of this CTA still hold the allocation open
  }

  free_.push_back({handle, it->second});
  allocs_.erase(it);
  cta_handle_.erase(cta_id);
  cta_dealloc_warps_.erase(cta_id);
  // Coalesce adjacent free ranges to keep the allocator from fragmenting.
  std::sort(free_.begin(), free_.end());
  for (size_t i = 0; i + 1 < free_.size();) {
    if (free_[i].first + free_[i].second == free_[i + 1].first) {
      free_[i].second += free_[i + 1].second;
      free_.erase(free_.begin() + i + 1);
    } else {
      ++i;
    }
  }
}

uint32_t TcuTmem::alloc_ncols(uint32_t handle) const {
  auto it = allocs_.find(handle);
  return (it == allocs_.end()) ? 0 : it->second;
}

///////////////////////////////////////////////////////////////////////////////

void TcuTmem::validate_lane_col(uint32_t lane, uint32_t col) const {
  if (lane >= kLanes) {
    std::cout << "Error: TMEM lane " << lane << " exceeds kLanes=" << kLanes << std::endl;
    std::abort();
  }
  for (auto& kv : allocs_) {
    if (col >= kv.first && col < kv.first + kv.second) return;
  }
  std::cout << "Error: TMEM column " << col << " not within any active allocation" << std::endl;
  std::abort();
}

uint32_t TcuTmem::read(uint32_t lane, uint32_t col) const {
  this->validate_lane_col(lane, col);
  return data_.at(lane).at(col);
}

void TcuTmem::write(uint32_t lane, uint32_t col, uint32_t value) {
  this->validate_lane_col(lane, col);
  data_.at(lane).at(col) = value;
}

///////////////////////////////////////////////////////////////////////////////
// Bank arbitration timing model

namespace {

// Build one bank's ARB_W-wide request vector, as req_vec = {ldst, compute}:
// compute at pool index bi, ldst at kBanks + bi.
BitVector<> build_req_vec(uint32_t bank,
                          const std::array<TcuTmem::BankReq, TcuTmem::kBanks>& compute,
                          const std::array<TcuTmem::BankReq, TcuTmem::kBanks>& ldst,
                          const std::bitset<TcuTmem::kBanks>& ldst_suppress) {
  BitVector<> req_vec(TcuTmem::kArbW);
  for (uint32_t bi = 0; bi < TcuTmem::kBanks; ++bi) {
    if (compute.at(bi).valid && TcuTmem::bank_of(compute.at(bi).lane_base) == bank) {
      req_vec.set(bi);
    }
    if (ldst.at(bi).valid && TcuTmem::bank_of(ldst.at(bi).lane_base) == bank
     && !ldst_suppress.test(bi)) {
      req_vec.set(TcuTmem::kBanks + bi);
    }
  }
  return req_vec;
}

} // namespace

void TcuTmem::arb_step(const CycleReqs& reqs) {
  // publish last cycle's registered read grants
  grants_ = Grants{};
  std::bitset<kBanks> ldst_rd_won;
  for (uint32_t bi = 0; bi < kBanks; ++bi) {
    uint32_t cmp_bank = bank_of(reqs.compute_rd.at(bi).lane_base);
    if (rd_grant_onehot_d_.at(cmp_bank).test(bi)) {
      grants_.compute_rd.set(bi);
    }
    uint32_t ldst_bank = bank_of(reqs.ldst_rd.at(bi).lane_base);
    if (rd_grant_onehot_d_.at(ldst_bank).test(kBanks + bi)) {
      ldst_rd_won.set(bi);
      grants_.ldst_rd.set(bi);
    }
  }

  // read arbitration
  std::array<std::bitset<kArbW>, kBanks> rd_grant_onehot{};
  for (uint32_t r = 0; r < kBanks; ++r) {
    auto req_vec = build_req_vec(r, reqs.compute_rd, reqs.ldst_rd, ldst_rd_won);
    if (req_vec.count() > 1) {
      ++perf_stats_.bank_stalls;
    }
    uint32_t winner = rd_arb_.at(r).grant(req_vec);
    if (winner != -1u) {
      rd_grant_onehot.at(r).set(winner);
    }
  }

  // write arbitration
  for (uint32_t r = 0; r < kBanks; ++r) {
    auto req_vec = build_req_vec(r, reqs.compute_wr, reqs.ldst_wr, ldst_wr_won_);
    if (req_vec.count() > 1) {
      ++perf_stats_.bank_stalls;
    }
    uint32_t winner = wr_arb_.at(r).grant(req_vec);
    if (winner == -1u)
      continue;
    // Writes are consumed the cycle they are granted, so route immediately.
    if (winner < kBanks) {
      uint32_t bi = winner;
      if (reqs.compute_wr.at(bi).valid) {
        grants_.compute_wr.set(bi);
      }
    } else {
      uint32_t bi = winner - kBanks;
      if (reqs.ldst_wr.at(bi).valid) {
        grants_.ldst_wr.set(bi);
        ldst_wr_won_.set(bi);
      }
    }
  }

  // register the read grant for next cycle
  rd_grant_onehot_d_ = rd_grant_onehot;
}

#endif // VX_CFG_TCU_TMEM_ENABLE
