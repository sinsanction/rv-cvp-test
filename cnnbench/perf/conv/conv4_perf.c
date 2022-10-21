#include <benchmark.h>
#include "cnn.h"

#define N N_SIZE
#define S 1

static uint8_t *A;
static uint64_t vwidth;
static int test_pass;
static int a_size;

inline int round_up_div(int a, int b) {
  int div = a / b;
  int rem = a % b;
  if (rem == 0) {
    return div;
  }
  else {
    return div + 1;
  }
}

void bench_conv4_perf_prepare() {
  bench_srand(1);
  vwidth = 0x2222222222222222;
  a_size = round_up_div(N * N, 2);
  A = (uint8_t *)bench_alloc(sizeof(uint8_t) * a_size);
  for (int i=0; i<a_size; i++) {
    A[i] = bench_rand() & 0xff;
  }
  test_pass = 1;
}

void bench_conv4_perf_run() {
  int k;              //kernel size
  int m;              //output size
  int32_t *B;         //cnn output
  int8_t *kernel;     //kernel
  int k_size;

  k = 5;
  m = (N - k) / S + 1;
  B = (int32_t *)bench_alloc(sizeof(int32_t) * m * m);
  k_size = round_up_div(k * k, 4);
  kernel = (int8_t *)bench_alloc(sizeof(int8_t) * k_size);

  for (int i=0; i<k_size; i++) {
    kernel[i] = bench_rand() & 0xff;
  }

  LoadV_Width((uint64_t)&vwidth);

  uint64_t kernel_ptr = ((uint64_t)kernel) << 2;
  LoadV_D_Kernel(kernel_ptr, k, 0, 0);
  LoadV_D_Kernel(kernel_ptr+k, k, 1, 0);
  LoadV_D_Kernel(kernel_ptr+2*k, k, 2, 0);
  LoadV_D_Kernel(kernel_ptr+3*k, k, 3, 0);
  LoadV_D_Kernel(kernel_ptr+4*k, k, 4, 0);

  uint64_t col_ptr;
  for (int i=0; i<m; i++) {
    col_ptr = (((uint64_t)A) << 1) + i;
    LoadV_D_Main((uint64_t)(col_ptr), k, 0, 0);
    LoadV_D_Main((uint64_t)(col_ptr+N), k, 1, 0);
    LoadV_D_Main((uint64_t)(col_ptr+2*N), k, 2, 0);
    LoadV_D_Main((uint64_t)(col_ptr+3*N), k, 3, 0);
    LoadV_D_Main((uint64_t)(col_ptr+4*N), k, 4, 0);
    col_ptr += 5*N;
    B[0 * m + i] = Conv(k);

    for (int j=1; j<m; j++) {
      LoadV_P((uint64_t)(col_ptr), k, 0);
      B[j * m + i] = Conv(k);
      col_ptr += N;
    }
  }

  bench_free(B);
  bench_free(kernel);
}

int bench_conv4_perf_validate() {
  bench_free(A);
  return (setting->checksum == 0x00020003) && test_pass;
}
