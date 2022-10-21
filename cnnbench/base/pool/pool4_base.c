#include <benchmark.h>
#include "cnn.h"

#define N N_SIZE
#define S 1

static uint8_t *A;
static int test_pass;
static int a_size;

inline uint8_t max(uint8_t a, uint8_t b) {
  return (a >= b) ? a : b;
}

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

void bench_pool4_base_prepare() {
  bench_srand(1);
  a_size = round_up_div(N * N, 2);
  A = (uint8_t *)bench_alloc(sizeof(uint8_t) * a_size);
  for (int i=0; i<a_size; i++) {
    A[i] = bench_rand() & 0xff;
  }
  test_pass = 1;
}

void bench_pool4_base_run() {
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
        tmp_res_max = max(tmp_res_max, get_main_uint4(A, (0 + sj) * N + (i + si)));
        tmp_res_avg += get_main_uint4(A, (0 + sj) * N + (i + si));
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
          tmp_res_max = max(tmp_res_max, get_main_uint4(A, (j + sj) * N + (i + si)));
          tmp_res_avg += get_main_uint4(A, (j + sj) * N + (i + si));
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

int bench_pool4_base_validate() {
  bench_free(A);
  return (setting->checksum == 0x00010013) && test_pass;
}
