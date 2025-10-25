#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto A = reinterpret_cast<int8_t*>(arg->A_addr);
	auto B = reinterpret_cast<int8_t*>(arg->B_addr);
	auto C = reinterpret_cast<int32_t*>(arg->C_addr);
    auto size = arg->size;

    int col = blockIdx.x;
    int row = blockIdx.y;

    int32_t sum(0);
    for (int k = 0; k < size; k += 4) {
        uint32_t packedA = *((int*)(A + row * size + k));
        uint32_t packedB = ((uint8_t)B[(k+0) * size + col] << 0)
                         | ((uint8_t)B[(k+1) * size + col] << 8)
                         | ((uint8_t)B[(k+2) * size + col] << 16)
                         | ((uint8_t)B[(k+3) * size + col] << 24);
        sum += vx_dot8(packedA, packedB);
    }
    C[row * size + col] = sum;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
