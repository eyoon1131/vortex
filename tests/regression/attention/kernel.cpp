#include <vx_spawn.h>
#include "common.h"
#include <cmath>
#include <algorithm>

void kernel0_body(kernel_arg_t* __UNIFORM__ arg) {
	auto Q = reinterpret_cast<TYPE*>(arg->Q_addr);
	auto K = reinterpret_cast<TYPE*>(arg->K_addr);
	auto S = reinterpret_cast<TYPE*>(arg->S_addr);
    auto N = arg->N;
    auto d = arg->d;

    int col = blockIdx.x;
    int row = blockIdx.y;

    if (row < N && col < N) {
        TYPE sum(0);
        for (int e = 0; e < d; ++e) {
            sum += Q[row * d + e] * K[e * N + col];
        }
        S[row * N + col] = sum;
    }
}

void kernel1_body(kernel_arg_t* __UNIFORM__ arg) {
	auto S = reinterpret_cast<TYPE*>(arg->S_addr);
	auto P = reinterpret_cast<TYPE*>(arg->P_addr);
    auto N = arg->N;

    int row = blockIdx.x;

    TYPE max_val = S[row * N];
    for (uint32_t col = 1; col < N; ++col) {
      max_val = std::max(max_val, S[row * N + col]);
    }

    TYPE local_P[N];
    TYPE exp_sum = 0;
    for (uint32_t col = 0; col < N; ++col) {
      auto exp = std::exp(S[row * N + col] - max_val);
      local_P[col] = exp;
      exp_sum += exp;
    }

    for (uint32_t col = 0; col < N; ++col) {
      P[row * N + col] = local_P[col] / exp_sum;
    }

}

void kernel2_body(kernel_arg_t* __UNIFORM__ arg) {
	auto P = reinterpret_cast<TYPE*>(arg->P_addr);
	auto V = reinterpret_cast<TYPE*>(arg->V_addr);
	auto O = reinterpret_cast<TYPE*>(arg->O_addr);
    auto N = arg->N;
    auto d = arg->d;

    int col = blockIdx.x;
    int row = blockIdx.y;

    if (row < N && col < d) {
        TYPE sum(0);
        for (int e = 0; e < N; ++e) {
            sum += P[row * N + e] * V[e * d + col];
        }
        O[row * d + col] = sum;
    }
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    if (arg->kernel_id == 0)
	    return vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel0_body, arg);
    if (arg->kernel_id == 1)
	    return vx_spawn_threads(1, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel1_body, arg);
    if (arg->kernel_id == 2)
	    return vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel2_body, arg);
}
