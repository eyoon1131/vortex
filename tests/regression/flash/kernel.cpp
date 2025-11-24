#include <vx_spawn.h>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include "common.h"

void kernel_body(kernel_arg_t *arg) {
	// Setup buffer arguments
  auto Q_ptr = reinterpret_cast<TYPE*>(arg->Q_addr);
  auto K_ptr = reinterpret_cast<TYPE*>(arg->K_addr);
  auto V_ptr = reinterpret_cast<TYPE*>(arg->V_addr);
  auto O_ptr = reinterpret_cast<TYPE*>(arg->O_addr);

  auto seq_len = arg->seq_len;
  auto head_dim = arg->head_dim;
  auto block_size_r = arg->block_size_r;
  auto block_size_c = arg->block_size_c;

  // Allocate local memory
  // Must store tile of Q (b_r x d) +  K, V (b_c x d)
	auto local_ptr = __local_mem((block_size_r + 2 * block_size_c) * head_dim * sizeof(TYPE));
  auto local_Q = (TYPE*)local_ptr;
  auto local_K = (TYPE*)local_Q + block_size_r * head_dim;
  auto local_V = (TYPE*)local_K + block_size_c * head_dim;
  // auto local_V = (TYPE*)local_K + block_size_c * head_dim;
  // auto local_l = (TYPE*)local_V + block_size_c * head_dim;
  // auto local_m = (TYPE*)local_l + block_size_r;

  // Determine global row index
  auto g_row = blockIdx.x * blockDim.x + threadIdx.x;

  // Determine local row index
  auto l_row = threadIdx.x;

  // Load Q_i from HBM
  for (uint32_t col = 0; col < head_dim; ++col)
    local_Q[l_row * head_dim + col] = Q_ptr[g_row * head_dim + col];
  
  // Initialize O_i
  TYPE local_O[head_dim];
  for (uint32_t col = 0; col < head_dim; ++col)
    local_O[col] = 0;

  // Initialize m_i, l_i
  TYPE local_m(std::numeric_limits<TYPE>::lowest());
  TYPE local_l(0);


  // Loop over blocks of K and V
  for (uint32_t j = 0; j < seq_len; j += block_size_c) {
    // Load K_j and V_j^T from HBM
    // block_size_c % block_size_r = 0
    for (uint32_t k = 0; k < block_size_c / block_size_r; ++k) {
      uint32_t row = k * block_size_r + l_row;
      for (uint32_t col = 0; col < head_dim; ++col) {
        local_K[row * head_dim + col] = K_ptr[(j + row) * head_dim + col];
        // Store transpose of V_j
        local_V[col * block_size_c + row] = V_ptr[(j + row) * head_dim + col];
      }
    }

    // Synchronize all warps in current group
    __syncthreads();

    // Compute S_ij = Q_i * K_j^T
    TYPE local_S[block_size_c];
    for (uint32_t k = 0; k < block_size_c; ++k) {
      // Compute the dot product of Q row and K tile
      auto Q_i = (TYPE*)local_Q + l_row * head_dim;
      auto K_j = (TYPE*)local_K + k * head_dim;
      local_S[k] = std::inner_product(Q_i, Q_i + head_dim, K_j, TYPE(0));
    }

    // Compute rowmax
    TYPE rowmax(*std::max_element(local_S, local_S + block_size_c));

    // Compute P_ij and rowsum
    TYPE local_P[block_size_c];
    TYPE rowsum(0);
    for (uint32_t k = 0; k < block_size_c; ++k) {
      local_P[k] = std::exp(local_S[k] - rowmax);
      rowsum += local_P[k];
    }

    // Compute new local_m and local_l
    TYPE new_m(std::max(local_m, rowmax));
    TYPE new_l(std::exp(local_m - new_m) * local_l + std::exp(rowmax - new_m) * rowsum);

    // Update O
    for (uint32_t col = 0; col < head_dim; ++col) {
      // Old contribution
      TYPE old_O(std::exp(local_m - new_m) * local_O[col]);
      // New contribution
      auto V_j = (TYPE*)local_V + col * block_size_c;
      TYPE dot(std::inner_product(local_P, local_P + block_size_c, V_j, TYPE(0)));
      TYPE new_O(std::exp(rowmax - new_m) * dot);
      // Update O
      local_O[col] = old_O + new_O;
    }

    // Update m and l for next iteration
    local_m = new_m;
    local_l = new_l;

    // Synchronize all warps in current group
    __syncthreads();
  }

  // Write O back to HBM
  for (uint32_t col = 0; col < head_dim; ++col) {
    O_ptr[g_row * head_dim + col] = local_O[col] / local_l;
  }
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
