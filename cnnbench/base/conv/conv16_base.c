#include <benchmark.h>
#include "cnn.h"

#define N N_SIZE
#define S 1

static uint16_t A[N][N];
static int test_pass;

void bench_conv16_base_prepare() {
  bench_srand(1);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = bench_rand() & 0xffff;
    }
  }
  test_pass = 1;
}

void bench_conv16_base_run() {
  int k;              //kernel size
  int m;              //output size
  int32_t *C;        //std output
  int8_t *kernel;     //kernel

  k = 5;
  m = (N - k) / S + 1;
  C = (int32_t *)bench_alloc(sizeof(int32_t) * m * m);
  kernel = (int8_t *)bench_alloc(sizeof(int8_t) * k * k);

  for (int i=0; i<k*k; i++) {
    kernel[i] = bench_rand() & 0xff;
  }

  for (int i=0; i<m; i++) {
    // std res
    int32_t tmp_res = 0;
    for (int si=0; si<k; si++) {
      for (int sj=0; sj<k; sj++) {
        tmp_res += A[0 + sj][i + si] * kernel[si * k + sj];
      }
    }
    C[0 * m + i] = tmp_res;//(tmp_res < 0) ? 0 : (tmp_res > 0xffff) ? 0xffff : tmp_res;

    for (int j=1; j<m; j++) {
      // std res
      int32_t tmp_res = 0;
      for (int si=0; si<k; si++) {
        for (int sj=0; sj<k; sj++) {
          tmp_res += A[j + sj][i + si] * kernel[si * k + sj];
        }
      }
      C[j * m + i] = tmp_res;//(tmp_res < 0) ? 0 : (tmp_res > 0xffff) ? 0xffff : tmp_res;
    }
  }

  bench_free(C);
  bench_free(kernel);

}

int bench_conv16_base_validate() {
  return (setting->checksum == 0x00010001) && test_pass;
}
