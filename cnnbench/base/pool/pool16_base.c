#include <benchmark.h>
#include "cnn.h"

#define N N_SIZE
#define S 1

static uint16_t A[N][N];
static int test_pass;

inline uint16_t max(uint16_t a, uint16_t b) {
  return (a >= b) ? a : b;
}

void bench_pool16_base_prepare() {
  bench_srand(1);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = bench_rand() & 0xffff;
    }
  }
  test_pass = 1;
}

void bench_pool16_base_run() {
  int k;               //kernel size
  int m;               //output size
  uint16_t *Cm;        //std max output
  uint16_t *Ca;        //std avg output

  k = 5;
  m = (N - k) / S + 1;
  Cm = (uint16_t *)bench_alloc(sizeof(uint16_t) * m * m);
  Ca = (uint16_t *)bench_alloc(sizeof(uint16_t) * m * m);

  for (int i=0; i<m; i++) {
    // std res
    uint16_t tmp_res_max = 0;
    uint32_t tmp_res_avg = 0;
    for (int si=0; si<k; si++) {
      for (int sj=0; sj<k; sj++) {
        tmp_res_max = max(tmp_res_max, A[0 + sj][i + si]);
        tmp_res_avg += A[0 + sj][i + si];
      }
    }
    Cm[0 * m + i] = tmp_res_max;
    uint32_t div = tmp_res_avg / (k * k);
    uint32_t rem = tmp_res_avg % (k * k);
    uint32_t cin = ((rem * 2) >= (k * k)) ? 1 : 0;
    Ca[0 * m + i] = div + cin;

    for (int j=1; j<m; j++) {
      // std res
      uint16_t tmp_res_max = 0;
      uint32_t tmp_res_avg = 0;
      for (int si=0; si<k; si++) {
        for (int sj=0; sj<k; sj++) {
          tmp_res_max = max(tmp_res_max, A[j + sj][i + si]);
          tmp_res_avg += A[j + sj][i + si];
        }
      }
      Cm[j * m + i] = tmp_res_max;
      uint32_t div = tmp_res_avg / (k * k);
      uint32_t rem = tmp_res_avg % (k * k);
      uint32_t cin = ((rem * 2) >= (k * k)) ? 1 : 0;
      Ca[j * m + i] = div + cin;
    }
  }

  bench_free(Cm);
  bench_free(Ca);

}

int bench_pool16_base_validate() {
  return (setting->checksum == 0x00010011) && test_pass;
}
