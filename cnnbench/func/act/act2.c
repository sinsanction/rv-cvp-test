#include <benchmark.h>
#include "cnn.h"

#define N N_SIZE
#define S 1

static uint8_t *A;
static uint64_t vwidth;
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

void bench_act2_prepare() {
  bench_srand(1);
  vwidth = 0x1111111111111111;
  a_size = round_up_div(N * N, 4);
  A = (uint8_t *)bench_alloc(sizeof(uint8_t) * a_size);
  for (int i=0; i<a_size; i++) {
    A[i] = bench_rand() & 0xff;
  }
  test_pass = 1;
}

void bench_act2_run() {
  int m;             //output size
  uint8_t *B;        //act output
  uint8_t *C;        //std output
  int pass;

  m = N;
  B = (uint8_t *)bench_alloc(sizeof(uint8_t) * m * m);
  C = (uint8_t *)bench_alloc(sizeof(uint8_t) * m * m);

  assert(((uint64_t)A % sizeof(uint64_t)) == 0);
  assert(((uint64_t)B % sizeof(uint64_t)) == 0);
  assert(((uint64_t)C % sizeof(uint64_t)) == 0);

  LoadV_Width((uint64_t)&vwidth);

  int num = sizeof(uint64_t) / sizeof(uint8_t) * 4;
  int size = round_up_div((m * m), num);
  uint64_t zero;
  uint64_t *a_ptr;
  uint64_t *b_ptr;
  uint64_t *c_ptr;

  printf("\nzero=0 begin: \n");

  zero = 0x0;
  a_ptr = (uint64_t *)A;
  b_ptr = (uint64_t *)B;
  c_ptr = (uint64_t *)C;
  pass = 1;
  for (int i=0; i<size; i++) {
    b_ptr[i] = Act(a_ptr[i], zero);

    uint64_t tmp_res = 0;
    for (int j=0; j<num; j++) {
      tmp_res |= (uint64_t)max(get_main_uint2(A, i * num + j), 0x0) << (j * 2);
    }
    c_ptr[i] = tmp_res;

    if (b_ptr[i] != c_ptr[i]) {
      printf("  act error: i=%d, act_res=%lx, std_res=%lx, tmp_res=%lx\n", i, b_ptr[i], c_ptr[i], tmp_res);
      for (int sj=0; sj<num; sj++) {
        printf("  %d(%x)", get_main_uint2(A, i * num + sj), get_main_uint2(A, i * num + sj));
      }
      printf("\n");
      pass = 0;
    }
    else {
      ;//printf("  ok: i=%d, std_res=%lx, tmp_res=%lx\n", i, b_ptr[i], tmp_res);
    }
  }
  if (pass == 1) {
    printf("zero=0 end: pass!!!\n");
  }
  else {
    printf("zero=0 end: fail\n");
    test_pass = 0;
  }

  printf("\nzero=2 begin: \n");

  zero = 0xaaaaaaaaaaaaaaaa;
  a_ptr = (uint64_t *)A;
  b_ptr = (uint64_t *)B;
  c_ptr = (uint64_t *)C;
  pass = 1;
  for (int i=0; i<size; i++) {
    b_ptr[i] = Act(a_ptr[i], zero);

    uint64_t tmp_res = 0;
    for (int j=0; j<num; j++) {
      tmp_res |= (uint64_t)max(get_main_uint2(A, i * num + j), 0x2) << (j * 2);
    }
    c_ptr[i] = tmp_res;

    if (b_ptr[i] != c_ptr[i]) {
      printf("  act error: i=%d, act_res=%lx, std_res=%lx, tmp_res=%lx\n", i, b_ptr[i], c_ptr[i], tmp_res);
      for (int sj=0; sj<num; sj++) {
        printf("  %d(%x)", get_main_uint2(A, i * num + sj), get_main_uint2(A, i * num + sj));
      }
      printf("\n");
      pass = 0;
    }
    else {
      ;//printf("  ok: i=%d, std_res=%lx, tmp_res=%lx\n", i, b_ptr[i], tmp_res);
      /*for (int sj=0; sj<num; sj++) {
        printf("  %d(%x)", a_ptr[sj], a_ptr[sj]);
      }
      printf("\n");*/
    }
  }
  if (pass == 1) {
    printf("zero=2 end: pass!!!\n");
  }
  else {
    printf("zero=2 end: fail\n");
    test_pass = 0;
  }

  bench_free(B);
  bench_free(C);
}

int bench_act2_validate() {
  bench_free(A);
  return (setting->checksum == 0x00000024) && test_pass;
}
