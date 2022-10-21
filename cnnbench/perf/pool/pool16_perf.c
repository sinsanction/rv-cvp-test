#include <benchmark.h>
#include "cnn.h"

#define N N_SIZE
#define S 1

static uint16_t A[N][N];
static uint64_t vwidth;
static int test_pass;

void bench_pool16_perf_prepare() {
  bench_srand(1);
  vwidth = 0x8888888888888888;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = bench_rand() & 0xffff;
    }
  }
  test_pass = 1;
}

void bench_pool16_perf_run() {
  int k;               //kernel size
  int m;               //output size
  uint16_t *Bm;        //pool max output
  uint16_t *Ba;        //pool avg output

  k = 5;
  m = (N - k) / S + 1;
  Bm = (uint16_t *)bench_alloc(sizeof(uint16_t) * m * m);
  Ba = (uint16_t *)bench_alloc(sizeof(uint16_t) * m * m);

  LoadV_Width((uint64_t)&vwidth);

  uint16_t *col_ptr;
  for (int i=0; i<m; i++) {
    col_ptr = &A[0][i];
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

int bench_pool16_perf_validate() {
  return (setting->checksum == 0x00020011) && test_pass;
}
