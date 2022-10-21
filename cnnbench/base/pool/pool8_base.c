#include <benchmark.h>
#include "cnn.h"

#define N N_SIZE
#define S 1

static uint8_t A[N][N];
static int test_pass;

inline uint8_t max(uint8_t a, uint8_t b) {
  return (a >= b) ? a : b;
}

void bench_pool8_base_prepare() {
  bench_srand(1);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = bench_rand() & 0xff;
    }
  }
  test_pass = 1;
}

void bench_pool8_base_run() {
  int k;              //kernel size
  int m;              //output size
  uint8_t *Cm;        //std max output
  uint8_t *Ca;        //std avg output

  k = 5;
  m = (N - k) / S + 1;
  Cm = (uint8_t *)bench_alloc(sizeof(uint8_t) * m * m);
  Ca = (uint8_t *)bench_alloc(sizeof(uint8_t) * m * m);

  for (int i=0; i<m; i++) {
    // std res
    uint8_t tmp_res_max = 0;
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
      uint8_t tmp_res_max = 0;
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

int bench_pool8_base_validate() {
  return (setting->checksum == 0x00010012) && test_pass;
}
