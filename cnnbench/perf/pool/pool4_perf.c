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

void bench_pool4_perf_prepare() {
  bench_srand(1);
  vwidth = 0x2222222222222222;
  a_size = round_up_div(N * N, 2);
  A = (uint8_t *)bench_alloc(sizeof(uint8_t) * a_size);
  for (int i=0; i<a_size; i++) {
    A[i] = bench_rand() & 0xff;
  }
  test_pass = 1;
}

void bench_pool4_perf_run() {
  int k;              //kernel size
  int m;              //output size
  uint8_t *Bm;        //pool max output
  uint8_t *Ba;        //pool avg output

  k = 5;
  m = (N - k) / S + 1;
  Bm = (uint8_t *)bench_alloc(sizeof(uint8_t) * m * m);
  Ba = (uint8_t *)bench_alloc(sizeof(uint8_t) * m * m);

  LoadV_Width((uint64_t)&vwidth);

  uint64_t col_ptr;
  for (int i=0; i<m; i++) {
    col_ptr = (((uint64_t)A) << 1) + i;
    LoadV_D_Main((uint64_t)(col_ptr), k, 0, 0);
    LoadV_D_Main((uint64_t)(col_ptr+N), k, 1, 0);
    LoadV_D_Main((uint64_t)(col_ptr+2*N), k, 2, 0);
    LoadV_D_Main((uint64_t)(col_ptr+3*N), k, 3, 0);
    LoadV_D_Main((uint64_t)(col_ptr+4*N), k, 4, 0);
    col_ptr += 5*N;
    Bm[0 * m + i] = Pool_Max(k);
    Ba[0 * m + i] = Pool_Avg(k);

    for (int j=1; j<m; j++) {
      LoadV_P((uint64_t)(col_ptr), k, 0);
      Bm[j * m + i] = Pool_Max(k);
      Ba[j * m + i] = Pool_Avg(k);
      col_ptr += N;
    }
  }

  bench_free(Bm);
  bench_free(Ba);

}

int bench_pool4_perf_validate() {
  bench_free(A);
  return (setting->checksum == 0x00020013) && test_pass;
}
