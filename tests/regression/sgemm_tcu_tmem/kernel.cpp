#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;

using ctx = vt::umma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE, UMMA_NRC>;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  uint32_t tid = threadIdx.x;
  uint32_t num_threads = blockDim.x;  // warps * NUM_THREADS
  uint32_t warp_rank = tid / NUM_THREADS; // warp_rank is not guaranteed to equal wid set by device (vx_warp_id())
  uint32_t num_warps = num_threads / NUM_THREADS;

  // CTA tile dimensions
  uint32_t cta_M = num_warps * ctx::xtileM;
  uint32_t tile_row = blockIdx.y * cta_M;
  uint32_t tile_col = blockIdx.x * ctx::xtileN;

  // Shared memory layout: A [cta_M × tileK] then B [tileK × per_warp_N]
  auto smem   = reinterpret_cast<ctx::input_t *>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + cta_M * ctx::tileK;

  // ── TMEM allocation ───────────────────────────────────────────────────────
  // One elected thread allocates TMEM for the CTA
  if (warp_rank == 0 && tid == 0) {
    vt::vx_tmem_alloc(ctx::xtileN);
  }
  __syncthreads();

  // Initialize accumulator in TMEM to zero
  // Executed cooperatively by warps
  ctx::fill_tmem(0);
  __syncthreads();

  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    // Cooperatively load A [cta_M × tileK] into smem
    uint32_t a_size = cta_M * ctx::tileK;
    for (uint32_t i = tid; i < a_size; i += num_threads) {
      uint32_t r = i / ctx::tileK;
      uint32_t c = i % ctx::tileK;
      A_smem[r * ctx::tileK + c] = pA[(tile_row + r) * K + (k + c)];
    }

    // Cooperatively load B [tileK × per_warp_N] into smem
    uint32_t b_size = ctx::tileK * ctx::xtileN;
    for (uint32_t i = tid; i < b_size; i += num_threads) {
      uint32_t r = i / ctx::xtileN;
      uint32_t c = i % ctx::xtileN;
      B_smem[r * ctx::xtileN + c] = pB[(k + r) * N + (tile_col + c)];
    }

    __syncthreads();

    // Get each warp's A slice
    auto A_warp = ctx::warp_A_ptr(A_smem);
    auto desc_a = vt::vx_make_smem_desc(A_warp, ctx::tileK * sizeof(ctx::input_t));
    auto desc_b = vt::vx_make_smem_desc(B_smem, ctx::xtileN * sizeof(ctx::input_t));

    ctx::umma_sync(desc_a, desc_b);

    __syncthreads();
  }

  // ── Epilogue: TMEM → global memory ───────────────────────────────────────
  ctx::store_output(pC, tile_row, tile_col, N);
  __syncthreads();

  // ── TMEM deallocation ─────────────────────────────────────────────────────
  if (warp_rank == 0 && tid == 0) {
    vt::vx_tmem_dealloc();
  }
}
