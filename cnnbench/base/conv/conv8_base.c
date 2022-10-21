#include <benchmark.h>
#include "cnn.h"

#define N N_SIZE
#define S 1

static uint8_t A[N][N];
static int test_pass;

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

inline int8_t get_kernel_int4(int8_t *kernel, int i) {
  int j = i / 2;
  int r = i % 2;
  int8_t kernel_i = (kernel[j] >> (r * 4)) & 0xf;
  if (kernel_i & 0x8) {
    kernel_i = kernel_i | 0xf0;
  }
  return kernel_i;
}

void bench_conv8_base_prepare() {
  bench_srand(1);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = bench_rand() & 0xff;
    }
  }
  test_pass = 1;
}

void bench_conv8_base_run() {
  int k;              //kernel size
  int m;              //output size
  int32_t *C;         //std output
  int8_t *kernel;     //kernel
  int k_size;

  k = 5;
  m = (N - k) / S + 1;
  C = (int32_t *)bench_alloc(sizeof(int32_t) * m * m);
  k_size = round_up_div(k * k, 2);
  kernel = (int8_t *)bench_alloc(sizeof(int8_t) * k_size);

  for (int i=0; i<k_size; i++) {
    kernel[i] = bench_rand() & 0xff;
  }

  for (int i=0; i<m; i++) {
    // std res
    int32_t tmp_res = 0;
    for (int si=0; si<k; si++) {
      for (int sj=0; sj<k; sj++) {
        tmp_res += A[0 + sj][i + si] * get_kernel_int4(kernel, si * k + sj);
      }
    }
    C[0 * m + i] = tmp_res;//(tmp_res < 0) ? 0 : (tmp_res > 0xff) ? 0xff : tmp_res;

    for (int j=1; j<m; j++) {
      // std res
      int32_t tmp_res = 0;
      for (int si=0; si<k; si++) {
        for (int sj=0; sj<k; sj++) {
          tmp_res += A[j + sj][i + si] * get_kernel_int4(kernel, si * k + sj);
        }
      }
      C[j * m + i] = tmp_res;//(tmp_res < 0) ? 0 : (tmp_res > 0xff) ? 0xff : tmp_res;
    }
  }

  bench_free(C);
  bench_free(kernel);

}

int bench_conv8_base_validate() {
  return (setting->checksum == 0x00010002) && test_pass;
}
