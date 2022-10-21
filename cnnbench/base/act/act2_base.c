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

inline uint8_t get_main_uint2(uint8_t *data, int i) {
  int j = i / 4;
  int r = i % 4;
  uint8_t data_i = (data[j] >> (r * 2)) & 0x3;
  return data_i;
}

void bench_act2_base_prepare() {
  bench_srand(1);
  a_size = round_up_div(N * N, 4);
  A = (uint8_t *)bench_alloc(sizeof(uint8_t) * a_size);
  for (int i=0; i<a_size; i++) {
    A[i] = bench_rand() & 0xff;
  }
  test_pass = 1;
}

void bench_act2_base_run() {
  int m;             //output size
  uint8_t *C;        //std output

  m = N;
  C = (uint8_t *)bench_alloc(sizeof(uint8_t) * m * m);

  assert(((uint64_t)A % sizeof(uint64_t)) == 0);
  assert(((uint64_t)C % sizeof(uint64_t)) == 0);

  int num = sizeof(uint64_t) / sizeof(uint8_t) * 4;
  int size = round_up_div((m * m), num);
  uint64_t *c_ptr;

  c_ptr = (uint64_t *)C;
  for (int i=0; i<size; i++) {
    uint64_t tmp_res = 0;
    for (int j=0; j<num; j++) {
      tmp_res |= (uint64_t)max(get_main_uint2(A, i * num + j), 0x0) << (j * 2);
    }
    c_ptr[i] = tmp_res;
  }

  c_ptr = (uint64_t *)C;
  for (int i=0; i<size; i++) {
    uint64_t tmp_res = 0;
    for (int j=0; j<num; j++) {
      tmp_res |= (uint64_t)max(get_main_uint2(A, i * num + j), 0x2) << (j * 2);
    }
    c_ptr[i] = tmp_res;
  }

  bench_free(C);
}

int bench_act2_base_validate() {
  bench_free(A);
  return (setting->checksum == 0x00010024) && test_pass;
}
