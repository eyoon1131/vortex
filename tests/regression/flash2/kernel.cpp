#include <vx_spawn.h>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include "common.h"
// 26
static constexpr uint32_t HEAD_DIM = 8;
static constexpr uint32_t BLOCK_SIZE_C = 4;

void kernel_body(kernel_arg_t *arg) {
    // Setup buffer arguments
    float* Q_ptr = reinterpret_cast<float*>(arg->Q_addr);
    float* K_ptr = reinterpret_cast<float*>(arg->K_addr);
    float* V_ptr = reinterpret_cast<float*>(arg->V_addr);
    float* O_ptr = reinterpret_cast<float*>(arg->O_addr);

    auto seq_len = arg->seq_len;
    auto block_size_r = arg->block_size_r;

    // Allocate local memory
    // Must store tile of Q (b_r x d) +  K, V (b_c x d)
    auto local_ptr = __local_mem((block_size_r + 2 * BLOCK_SIZE_C) * HEAD_DIM * sizeof(float));
    auto local_Q = (float*)local_ptr;
    auto local_K = (float*)local_Q + block_size_r * HEAD_DIM;
    auto local_V = (float*)local_K + BLOCK_SIZE_C * HEAD_DIM;

    // Determine global/local row index
    auto g_row = blockIdx.x * blockDim.x + threadIdx.x;
    auto l_row = threadIdx.x;

    auto g_row_offset = g_row * HEAD_DIM;
    auto l_row_offset = l_row * HEAD_DIM;

    // Load Q_i from HBM
    #pragma clang loop unroll(full)
    for (uint32_t col = 0; col < HEAD_DIM; ++col)
        local_Q[l_row_offset + col] = Q_ptr[g_row_offset + col];

    // Initialize O_i in registers
    float O[HEAD_DIM];
    #pragma clang loop unroll(full)
    for (uint32_t col = 0; col < HEAD_DIM; ++col)
      O[col] = 0.0f;

    // 
    float sp_buf[BLOCK_SIZE_C];

    // Softmax accumulators
    float m = -INFINITY;
    float l = 0.f;

    for (uint32_t j = 0; j < seq_len; j += BLOCK_SIZE_C) {

        uint32_t block_offset = j * HEAD_DIM;

        // Load K_j and V_j^T
        // #pragma clang loop unroll(full)
        // for (int r = 0; r < 4; r++) {
        //     #pragma clang loop unroll(full)
        //     for (int c = 0; c < BLOCK_SIZE_C; c++) {
        //         float kval = K_ptr[block_offset + r*4 + c];
        //         local_K[r*4 + c] = kval;

        //         float vval = V_ptr[block_offset + r*4 + c];
        //         local_V[c*4 + r] = vval; // transpose V
        //     }
        // }
        // #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < BLOCK_SIZE_C / block_size_r; ++k) {
          auto row = k * block_size_r + l_row;
          auto row_offset = row * HEAD_DIM;
          // #pragma clang loop unroll(full)
          for (uint32_t col = 0; col < HEAD_DIM; ++col) {
            auto offset = row_offset + col;
            local_K[offset] = K_ptr[block_offset + offset];
            // Store transpose of V_j
            local_V[col * BLOCK_SIZE_C + row] = V_ptr[block_offset + offset];
          }
        }

        __syncthreads();

        float* Qi = local_Q + l_row*HEAD_DIM;

        // S_ij = Q_i · K_j^T
        #pragma clang loop unroll(full)
        for (int k = 0; k < BLOCK_SIZE_C; ++k) {
          // sp_buf[k] = 0;
          #pragma clang loop unroll(full)
          for (int elem = 0; elem < HEAD_DIM; ++elem)
            sp_buf[k] += Qi[elem] * local_K[k * HEAD_DIM + elem];
        }

        // Row max
        float rowmax = sp_buf[0];
        #pragma clang loop unroll(full)
        for (int k = 1; k < BLOCK_SIZE_C; ++k)
          if (sp_buf[k] > rowmax) rowmax = sp_buf[k];

        // Exps
        #pragma clang loop unroll(full)
        for (int k = 0; k < BLOCK_SIZE_C; ++k)
          sp_buf[k] = expf(sp_buf[k] - rowmax);

        float rowsum = 0;
        #pragma clang loop unroll(full)
        for (int k = 0; k < BLOCK_SIZE_C; ++k)
          rowsum += sp_buf[k];

        // Online softmax update
        float new_m = (m > rowmax ? m : rowmax);
        float new_l = expf(m - new_m) * l + expf(rowmax - new_m) * rowsum;

        float scale_old = expf(m - new_m);
        float scale_new = expf(rowmax - new_m);

        // Update O
        #pragma clang loop unroll(full)
        for (int k = 0; k < HEAD_DIM; ++k) {
          float dot = 0;
          #pragma clang loop unroll(full)
          for (int elem = 0; elem < BLOCK_SIZE_C; ++elem)
            dot += sp_buf[elem] * local_V[k * BLOCK_SIZE_C + elem];
          O[k] = scale_old * O[k] + scale_new * dot; 
        }

        m = new_m;
        l = new_l;

        // #pragma clang loop unroll(full)
        // for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k)
        //     sp_buf[k] = 0;

        // __syncthreads();
    }

    // Final normalization
    float inv_l = 1.0f / l;
    #pragma clang loop unroll(full)
    for (int k = 0; k < HEAD_DIM; ++k)
      O_ptr[g_row_offset + k] = O[k] * inv_l;
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
