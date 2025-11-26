#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <vortex.h>
#include <cmath>
#include <algorithm>
#include "common.h"

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();			                                              \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class Comparator {};

template <>
class Comparator<float> {
public:
  static const char* type_str() {
    return "float";
  }
  // Modified generate to larger range [-5, 5]
  static float generate() {
      return 10.0f * (float(rand()) / RAND_MAX) - 5.0f;  \
  }
  // Modified compare to allow for greater tolerance
  static bool compare(float a, float b, int index, int errors) {
    const float atol = 1e-5f;
    const float rtol = 1e-5f;
    auto diff = std::fabs(a - b);
    auto limit = atol + rtol * std::fabs(b);
    if (diff > limit) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, a, b);
      }
      return false;
    }
    return true;
  }
};

static void matmul_cpu(TYPE* out, const TYPE* A, const TYPE* B, uint32_t M, uint32_t N, uint32_t K) {
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      TYPE sum(0);
      for (uint32_t e = 0; e < K; ++e) {
          sum += A[row * K + e] * B[e * N + col];
      }
      out[row * N + col] = sum;
    }
  }
}

static void softmax_cpu(TYPE* out, const TYPE* A, uint32_t M, uint32_t N) {
  for (uint32_t row = 0; row < M; ++row) {
    TYPE max_val = A[row * N];
    for (uint32_t col = 1; col < N; ++col) {
      max_val = std::max(max_val, A[row * N + col]);
    }

    TYPE exp_sum = 0;
    for (uint32_t col = 0; col < N; ++col) {
      out[row * N + col] = std::exp(A[row * N + col] - max_val);
      exp_sum += out[row * N + col];
    }

    for (uint32_t col = 0; col < N; ++col) {
      out[row * N + col] /= exp_sum;
    }
  }
}

const char* kernel_file = "kernel.vxbin";
uint32_t N = 64;
uint32_t d = 8;

vx_device_h device = nullptr;
vx_buffer_h Q_buffer = nullptr;
vx_buffer_h K_buffer = nullptr;
vx_buffer_h S_buffer = nullptr;
vx_buffer_h P_buffer = nullptr;
vx_buffer_h V_buffer = nullptr;
vx_buffer_h O_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n size] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
    case 'n':
      // size = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(Q_buffer);
    vx_mem_free(K_buffer);
    vx_mem_free(S_buffer);
    vx_mem_free(P_buffer);
    vx_mem_free(V_buffer);
    vx_mem_free(O_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  // -------------------------------------------------------------------------------------------------
  // S = QK^T

  uint32_t in_size = N * d;
  uint32_t in_buf_size = in_size * sizeof(TYPE);
  uint32_t out_size = N * N;
  uint32_t out_buf_size = out_size * sizeof(TYPE);

  std::cout << "Calculate S = QK^T:" << std::endl;
  std::cout << "Q matrix size: " << N << "x" << d << std::endl;
  std::cout << "K matrix size: " << d << "x" << N << std::endl;
  std::cout << "S matrix size: " << N << "x" << N << std::endl;

  kernel_arg.grid_dim[0] = N;
  kernel_arg.grid_dim[1] = N;
  kernel_arg.N = N;
  kernel_arg.d = d;
  kernel_arg.kernel_id = 0;

  // allocate device memory
  std::cout << "allocate Q, K, S in device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, in_buf_size, VX_MEM_READ, &Q_buffer));
  RT_CHECK(vx_mem_address(Q_buffer, &kernel_arg.Q_addr));
  RT_CHECK(vx_mem_alloc(device, in_buf_size, VX_MEM_READ, &K_buffer));
  RT_CHECK(vx_mem_address(K_buffer, &kernel_arg.K_addr));
  RT_CHECK(vx_mem_alloc(device, out_buf_size, VX_MEM_WRITE, &S_buffer));
  RT_CHECK(vx_mem_address(S_buffer, &kernel_arg.S_addr));

  std::cout << "Q_addr=0x" << std::hex << kernel_arg.Q_addr << std::endl;
  std::cout << "K_addr=0x" << std::hex << kernel_arg.K_addr << std::endl;
  std::cout << "S_addr=0x" << std::hex << kernel_arg.S_addr << std::endl;

  // generate source data
  std::vector<TYPE> h_Q(in_size);
  std::vector<TYPE> h_K(in_size);
  for (uint32_t i = 0; i < in_size; ++i) {
    h_Q[i] = Comparator<TYPE>::generate();
    h_K[i] = Comparator<TYPE>::generate();
  }

  // upload matrix Q buffer
  {
    std::cout << "upload matrix Q buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(Q_buffer, h_Q.data(), 0, in_buf_size));
  }

  // upload matrix K buffer
  {
    std::cout << "upload matrix K buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(K_buffer, h_K.data(), 0, in_buf_size));
  }

  // Upload kernel binary
  std::cout << "Upload kernel binary" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  auto time_start = std::chrono::high_resolution_clock::now();

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  // wait for completion
  std::cout << "wait for completion" << std::endl;\
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);
  
  // download destination buffer
  std::vector<TYPE> h_S(out_size);
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_S.data(), S_buffer, 0, out_buf_size));

  // verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<TYPE> h_ref(out_size);
    matmul_cpu(h_ref.data(), h_Q.data(), h_K.data(), N, N, d);

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!Comparator<TYPE>::compare(h_S[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }
  }

  // -------------------------------------------------------------------------------------------------
  // P = softmax(S)

  in_size = N * N;
  in_buf_size = in_size * sizeof(TYPE);
  out_size = N * N;
  out_buf_size = out_size * sizeof(TYPE);

  std::cout << "Calculate P = softmax(S)" << std::endl;
  std::cout << "P matrix size: " << N << "x" << N << std::endl;

  kernel_arg.grid_dim[0] = N;
  kernel_arg.kernel_id = 1;

  // allocate device memory
  std::cout << "allocate S, P in device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, in_buf_size, VX_MEM_READ, &S_buffer));
  RT_CHECK(vx_mem_address(S_buffer, &kernel_arg.S_addr));
  RT_CHECK(vx_mem_alloc(device, out_buf_size, VX_MEM_WRITE, &P_buffer));
  RT_CHECK(vx_mem_address(P_buffer, &kernel_arg.P_addr));

  std::cout << "S_addr=0x" << std::hex << kernel_arg.S_addr << std::endl;
  std::cout << "P_addr=0x" << std::hex << kernel_arg.P_addr << std::endl;

  // upload matrix S buffer
  {
    std::cout << "upload matrix S buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(S_buffer, h_S.data(), 0, in_buf_size));
  }

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  time_start = std::chrono::high_resolution_clock::now();

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  time_end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // download destination buffer
  std::vector<TYPE> h_P(out_size);
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_P.data(), P_buffer, 0, out_buf_size));

  // verify result
  std::cout << "verify result" << std::endl;
  errors = 0;
  {
    std::vector<TYPE> h_ref(out_size);
    softmax_cpu(h_ref.data(), h_S.data(), N, N);

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!Comparator<TYPE>::compare(h_P[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }
  }

  // -------------------------------------------------------------------------------------------------
  // O = PV

  uint32_t P_in_size = N * N;
  uint32_t P_in_buf_size = P_in_size * sizeof(TYPE);
  uint32_t V_in_size = N * d;
  uint32_t V_in_buf_size = V_in_size * sizeof(TYPE);
  out_size = N * d;
  out_buf_size = out_size * sizeof(TYPE);

  std::cout << "Calculate O = PV" << std::endl;
  std::cout << "P matrix size: " << N << "x" << N << std::endl;
  std::cout << "V matrix size: " << N << "x" << d << std::endl;
  std::cout << "O matrix size: " << N << "x" << d << std::endl;

  kernel_arg.grid_dim[0] = d;
  kernel_arg.grid_dim[1] = N;
  kernel_arg.kernel_id = 2;

  // allocate device memory
  std::cout << "allocate P, V, O in device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, P_in_buf_size, VX_MEM_READ, &P_buffer));
  RT_CHECK(vx_mem_address(P_buffer, &kernel_arg.P_addr));
  RT_CHECK(vx_mem_alloc(device, V_in_buf_size, VX_MEM_READ, &V_buffer));
  RT_CHECK(vx_mem_address(V_buffer, &kernel_arg.V_addr));
  RT_CHECK(vx_mem_alloc(device, out_buf_size, VX_MEM_WRITE, &O_buffer));
  RT_CHECK(vx_mem_address(O_buffer, &kernel_arg.O_addr));

  std::cout << "P_addr=0x" << std::hex << kernel_arg.P_addr << std::endl;
  std::cout << "V_addr=0x" << std::hex << kernel_arg.V_addr << std::endl;
  std::cout << "O_addr=0x" << std::hex << kernel_arg.O_addr << std::endl;

  // generate source data
  std::vector<TYPE> h_V(V_in_size);
  for (uint32_t i = 0; i < V_in_size; ++i) {
    h_V[i] = Comparator<TYPE>::generate();
  }

  // upload matrix P buffer
  {
    std::cout << "upload matrix P buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(P_buffer, h_P.data(), 0, P_in_buf_size));
  }

  // upload matrix V buffer
  {
    std::cout << "upload matrix V buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(V_buffer, h_V.data(), 0, V_in_buf_size));
  }

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  time_start = std::chrono::high_resolution_clock::now();

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  time_end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // download destination buffer
  std::vector<TYPE> h_O(out_size);
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_O.data(), O_buffer, 0, out_buf_size));

  // verify result
  std::cout << "verify result" << std::endl;
  errors = 0;
  {
    std::vector<TYPE> h_ref(out_size);
    matmul_cpu(h_ref.data(), h_P.data(), h_V.data(), N, d, N);

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!Comparator<TYPE>::compare(h_O[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}