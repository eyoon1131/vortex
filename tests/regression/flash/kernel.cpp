#include <vx_spawn.h>
#include <cmath>
#include "common.h"

template<uint32_t HEAD_DIM, uint32_t BLOCK_SIZE_C>
void flash_kernel_body(kernel_arg_t *arg) {
    // Setup buffer arguments
    float* Q_ptr = reinterpret_cast<float*>(arg->Q_addr);
    float* K_ptr = reinterpret_cast<float*>(arg->K_addr);
    float* V_ptr = reinterpret_cast<float*>(arg->V_addr);
    float* O_ptr = reinterpret_cast<float*>(arg->O_addr);

    auto seq_len = arg->seq_len;
    auto block_size_r = arg->block_size_r;

    // Allocate local memory
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
    for (uint32_t k = 0; k < HEAD_DIM; ++k)
      local_Q[l_row_offset + k] = Q_ptr[g_row_offset + k];

    // Thread's row of Q block
    float* Q_row = local_Q + l_row * HEAD_DIM;

    // Initialize O_i
    float O_buf[HEAD_DIM];
    for (uint32_t k = 0; k < HEAD_DIM; ++k)
      O_buf[k] = 0.0f;

    // Create buffer to store row of S and P
    float sp_buf[BLOCK_SIZE_C];

    // Initialize m_i (rowmax), l_i (softmax denominator)
    float m = -INFINITY;
    float l = 0.0f;

    // Loop over blocks of K and V
    for (uint32_t j = 0; j < seq_len; j += BLOCK_SIZE_C) {
        auto block_offset = j * HEAD_DIM;

        // Load K_j and V_j^T
        for (uint32_t k = 0; k < BLOCK_SIZE_C / block_size_r; ++k) {
          auto row = k * block_size_r + l_row;
          auto row_offset = row * HEAD_DIM;
          for (uint32_t col = 0; col < HEAD_DIM; ++col) {
            auto offset = row_offset + col;
            local_K[offset] = K_ptr[block_offset + offset];
            local_V[col * BLOCK_SIZE_C + row] = V_ptr[block_offset + offset];
          }
        }

        __syncthreads();

        // Compute S_ij, the dot product of thread's Q row and each row of K_j
        for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k)
          sp_buf[k] = 0;
        for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k) 
          for (uint32_t elem = 0; elem < HEAD_DIM; ++elem)
            sp_buf[k] += Q_row[elem] * local_K[k * HEAD_DIM + elem];

        // Row max
        float rowmax = sp_buf[0];
        for (uint32_t k = 1; k < BLOCK_SIZE_C; ++k)
          rowmax = (sp_buf[k] > rowmax ? sp_buf[k] : rowmax);

        // Compute P_ij, the softmax numerators of S_ij
        for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k)
          sp_buf[k] = expf(sp_buf[k] - rowmax);

        // Row sum
        float rowsum = 0.0f;
        for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k)
          rowsum += sp_buf[k];

        // Compute new m and l
        float new_m = (m > rowmax ? m : rowmax);
        float new_l = expf(m - new_m) * l + expf(rowmax - new_m) * rowsum;

        // Weights of old and new O
        float old_weight = expf(m - new_m);
        float new_weight = expf(rowmax - new_m);

        // Update O
        for (uint32_t k = 0; k < HEAD_DIM; ++k) {
          // Compute dot product of thread's P_ij and each col of V_j
          float dot = 0.0f;
          for (uint32_t elem = 0; elem < BLOCK_SIZE_C; ++elem)
            dot += sp_buf[elem] * local_V[k * BLOCK_SIZE_C + elem]; 
          O_buf[k] = old_weight * O_buf[k] + new_weight * dot;
        }

        // Update m and l for next block
        m = new_m;
        l = new_l;

        __syncthreads();
    }

    // Normalize O by softmax denominator and write back to HBM
    float inv_l = 1.0f / l;
    for (uint32_t k = 0; k < HEAD_DIM; ++k)
      O_ptr[g_row_offset + k] = O_buf[k] * inv_l;
}

void flash_kernel_entry(kernel_arg_t* arg) {
  switch (arg->head_dim) {
    case 1:
      switch (arg->block_size_c) {
        case 8:
          flash_kernel_body<1,8>(arg);
          return;
        case 16:
          flash_kernel_body<1,16>(arg);
          return;
        case 32:
          flash_kernel_body<1,32>(arg);
          return;
        case 64:
          flash_kernel_body<1,64>(arg);
          return;
        case 128:
          flash_kernel_body<1,128>(arg);
          return;
      }
      break;
    case 2:
      switch (arg->block_size_c) {
        case 8:
          flash_kernel_body<2,8>(arg);
          return;
        case 16:
          flash_kernel_body<2,16>(arg);
          return;
        case 32:
          flash_kernel_body<2,32>(arg);
          return;
        case 64:
          flash_kernel_body<2,64>(arg);
          return;    
      }
      break;
    case 4:
      switch (arg->block_size_c) {
        case 8:
          flash_kernel_body<4,8>(arg);
          return;
        case 16:
          flash_kernel_body<4,16>(arg);
          return;
        case 32:
          flash_kernel_body<4,32>(arg);
          return;
      }
      break;
    case 8:
      switch (arg->block_size_c) {
        case 8:
          flash_kernel_body<8,8>(arg);
          return;
        case 16:
          flash_kernel_body<8,16>(arg);
          return;
      }
      break;
    case 16:
      switch (arg->block_size_c) {
        case 8:
          flash_kernel_body<16,8>(arg);
          return;
      }
      break;
    }
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)flash_kernel_entry, arg);
}
