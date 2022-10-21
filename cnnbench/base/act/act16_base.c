#include <benchmark.h>
#include "cnn.h"

#define N N_SIZE
#define S 1

static uint16_t A[N][N];
static int test_pass;

inline uint16_t max(uint16_t a, uint16_t b) {
  return (a >= b) ? a : b;
}

void bench_act16_base_prepare() {
  bench_srand(1);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = bench_rand() & 0xffff;
    }
  }
  test_pass = 1;
}

void bench_act16_base_run() {
  int m;              //output size
  uint16_t *C;        //std output

  m = N;
  C = (uint16_t *)bench_alloc(sizeof(uint16_t) * m * m);

  assert(((uint64_t)A % sizeof(uint64_t)) == 0);
  assert(((uint64_t)C % sizeof(uint64_t)) == 0);

  int num = sizeof(uint64_t) / sizeof(uint16_t);
  int size = (m * m) / num;
  uint16_t *a_ptr;
  uint64_t *c_ptr;

  a_ptr = (uint16_t *)A;
  c_ptr = (uint64_t *)C;
  for (int i=0; i<size; i++) {
    uint64_t tmp_res = 0;
    for (int j=0; j<num; j++) {
      tmp_res |= (uint64_t)max(a_ptr[j], 0x0) << (j * 16);
    }
    c_ptr[i] = tmp_res;
    a_ptr += num;
  }

  a_ptr = (uint16_t *)A;
  c_ptr = (uint64_t *)C;
  for (int i=0; i<size; i++) {
    uint64_t tmp_res = 0;
    for (int j=0; j<num; j++) {
      tmp_res |= (uint64_t)max(a_ptr[j], 0x4000) << (j * 16);
    }
    c_ptr[i] = tmp_res;
    a_ptr += num;
  }

  bench_free(C);
}

int bench_act16_base_validate() {
  return (setting->checksum == 0x00010021) && test_pass;
}
