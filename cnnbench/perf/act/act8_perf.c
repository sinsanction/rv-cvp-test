#include <benchmark.h>
#include "cnn.h"

#define N N_SIZE
#define S 1

static uint8_t A[N][N];
static uint64_t vwidth;
static int test_pass;

void bench_act8_perf_prepare() {
  bench_srand(1);
  vwidth = 0x4444444444444444;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = bench_rand() & 0xff;
    }
  }
  test_pass = 1;
}

void bench_act8_perf_run() {
  int m;             //output size
  uint8_t *B;        //act output

  m = N;
  B = (uint8_t *)bench_alloc(sizeof(uint8_t) * m * m);

  assert(((uint64_t)A % sizeof(uint64_t)) == 0);
  assert(((uint64_t)B % sizeof(uint64_t)) == 0);

  LoadV_Width((uint64_t)&vwidth);

  int num = sizeof(uint64_t) / sizeof(uint8_t);
  int size = (m * m) / num;
  uint64_t zero;
  uint8_t *a_ptr;
  uint64_t *b_ptr;

  zero = 0x0;
  a_ptr = (uint8_t *)A;
  b_ptr = (uint64_t *)B;
  for (int i=0; i<size; i++) {
    b_ptr[i] = Act(*(uint64_t *)a_ptr, zero);
    a_ptr += num;
  }

  zero = 0x8080808080808080;
  a_ptr = (uint8_t *)A;
  b_ptr = (uint64_t *)B;
  for (int i=0; i<size; i++) {
    b_ptr[i] = Act(*(uint64_t *)a_ptr, zero);
    a_ptr += num;
  }

  bench_free(B);
}

int bench_act8_perf_validate() {
  return (setting->checksum == 0x00020022) && test_pass;
}
