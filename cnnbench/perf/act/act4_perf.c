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

void bench_act4_perf_prepare() {
  bench_srand(1);
  vwidth = 0x2222222222222222;
  a_size = round_up_div(N * N, 2);
  A = (uint8_t *)bench_alloc(sizeof(uint8_t) * a_size);
  for (int i=0; i<a_size; i++) {
    A[i] = bench_rand() & 0xff;
  }
  test_pass = 1;
}

void bench_act4_perf_run() {
  int m;             //output size
  uint8_t *B;        //act output

  m = N;
  B = (uint8_t *)bench_alloc(sizeof(uint8_t) * m * m);

  assert(((uint64_t)A % sizeof(uint64_t)) == 0);
  assert(((uint64_t)B % sizeof(uint64_t)) == 0);

  LoadV_Width((uint64_t)&vwidth);

  int num = sizeof(uint64_t) / sizeof(uint8_t) * 2;
  int size = round_up_div((m * m), num);
  uint64_t zero;
  uint64_t *a_ptr;
  uint64_t *b_ptr;

  zero = 0x0;
  a_ptr = (uint64_t *)A;
  b_ptr = (uint64_t *)B;
  for (int i=0; i<size; i++) {
    b_ptr[i] = Act(a_ptr[i], zero);
  }

  zero = 0x8888888888888888;
  a_ptr = (uint64_t *)A;
  b_ptr = (uint64_t *)B;
  for (int i=0; i<size; i++) {
    b_ptr[i] = Act(a_ptr[i], zero);
  }

  bench_free(B);
}

int bench_act4_perf_validate() {
  bench_free(A);
  return (setting->checksum == 0x00020023) && test_pass;
}
