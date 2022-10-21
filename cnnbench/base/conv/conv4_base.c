#include <benchmark.h>
#include "cnn.h"

#define N N_SIZE
#define S 1

static uint8_t *A;
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

inline uint8_t get_main_uint4(uint8_t *data, int i) {
  int j = i / 2;
  int r = i % 2;
  uint8_t data_i = (data[j] >> (r * 4)) & 0xf;
  return data_i;
}

inline int8_t get_kernel_int2(int8_t *kernel, int i) {
  int j = i / 4;
  int r = i % 4;
  int8_t kernel_i = (kernel[j] >> (r * 2)) & 0x3;
  if (kernel_i & 0x2) {
    kernel_i = kernel_i | 0xfc;
  }
  return kernel_i;
}

void bench_conv4_base_prepare() {
  bench_srand(1);
  a_size = round_up_div(N * N, 2);
  A = (uint8_t *)bench_alloc(sizeof(uint8_t) * a_size);
  for (int i=0; i<a_size; i++) {
    A[i] = bench_rand() & 0xff;
  }
  test_pass = 1;
}

void bench_conv4_base_run() {
  int k;              //kernel size
  int m;              //output size
  int32_t *C;         //std output
  int8_t *kernel;     //kernel
  int k_size;

  k = 5;
  m = (N - k) / S + 1;
  C = (int32_t *)bench_alloc(sizeof(int32_t) * m * m);
  k_size = round_up_div(k * k, 4);
  kernel = (int8_t *)bench_alloc(sizeof(int8_t) * k_size);

  for (int i=0; i<k_size; i++) {
    kernel[i] = bench_rand() & 0xff;
  }

  for (int i=0; i<m; i++) {
    // std res
    int32_t tmp_res = 0;
    for (int si=0; si<k; si++) {
      for (int sj=0; sj<k; sj++) {
        tmp_res += get_main_uint4(A, (0 + sj) * N + (i + si)) * get_kernel_int2(kernel, si * k + sj);
      }
    }
    C[0 * m + i] = tmp_res;//(tmp_res < 0) ? 0 : (tmp_res > 0xf) ? 0xf : tmp_res;

    for (int j=1; j<m; j++) {
      // std res
      int32_t tmp_res = 0;
      for (int si=0; si<k; si++) {
        for (int sj=0; sj<k; sj++) {
          tmp_res += get_main_uint4(A, (j + sj) * N + (i + si)) * get_kernel_int2(kernel, si * k + sj);
        }
      }
      C[j * m + i] = tmp_res;//(tmp_res < 0) ? 0 : (tmp_res > 0xf) ? 0xf : tmp_res;
    }
  }

  bench_free(C);
  bench_free(kernel);

}

int bench_conv4_base_validate() {
  bench_free(A);
  return (setting->checksum == 0x00010003) && test_pass;
}
