#include <benchmark.h>
#include "cnn.h"

#define N N_SIZE
#define S 1

static uint8_t A[N][N];
static uint64_t vwidth;
static int test_pass;

inline uint8_t max(uint8_t a, uint8_t b) {
  return (a >= b) ? a : b;
}

void bench_act8_prepare() {
  bench_srand(1);
  vwidth = 0x4444444444444444;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = bench_rand() & 0xff;
    }
  }
  test_pass = 1;
}

void bench_act8_run() {
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

  int num = sizeof(uint64_t) / sizeof(uint8_t);
  int size = (m * m) / num;
  uint64_t zero;
  uint8_t *a_ptr;
  uint64_t *b_ptr;
  uint64_t *c_ptr;

  printf("\nzero=0 begin: \n");

  zero = 0x0;
  a_ptr = (uint8_t *)A;
  b_ptr = (uint64_t *)B;
  c_ptr = (uint64_t *)C;
  pass = 1;
  for (int i=0; i<size; i++) {
    b_ptr[i] = Act(*(uint64_t *)a_ptr, zero);

    uint64_t tmp_res = 0;
    for (int j=0; j<num; j++) {
      tmp_res |= (uint64_t)max(a_ptr[j], 0x0) << (j * 8);
    }
    c_ptr[i] = tmp_res;

    if (b_ptr[i] != c_ptr[i]) {
      printf("  act error: i=%d, act_res=%lx, std_res=%lx, tmp_res=%lx\n", i, b_ptr[i], c_ptr[i], tmp_res);
      for (int sj=0; sj<num; sj++) {
        printf("  %d(%x)", a_ptr[sj], a_ptr[sj]);
      }
      printf("\n");
      pass = 0;
    }
    else {
      ;//printf("  ok: i=%d, std_res=%lx, tmp_res=%lx\n", i, b_ptr[i], tmp_res);
    }

    a_ptr += num;
  }
  if (pass == 1) {
    printf("zero=0 end: pass!!!\n");
  }
  else {
    printf("zero=0 end: fail\n");
    test_pass = 0;
  }

  printf("\nzero=128 begin: \n");

  zero = 0x8080808080808080;
  a_ptr = (uint8_t *)A;
  b_ptr = (uint64_t *)B;
  c_ptr = (uint64_t *)C;
  pass = 1;
  for (int i=0; i<size; i++) {
    b_ptr[i] = Act(*(uint64_t *)a_ptr, zero);

    uint64_t tmp_res = 0;
    for (int j=0; j<num; j++) {
      tmp_res |= (uint64_t)max(a_ptr[j], 0x80) << (j * 8);
    }
    c_ptr[i] = tmp_res;

    if (b_ptr[i] != c_ptr[i]) {
      printf("  act error: i=%d, act_res=%lx, std_res=%lx, tmp_res=%lx\n", i, b_ptr[i], c_ptr[i], tmp_res);
      for (int sj=0; sj<num; sj++) {
        printf("  %d(%x)", a_ptr[sj], a_ptr[sj]);
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

    a_ptr += num;
  }
  if (pass == 1) {
    printf("zero=128 end: pass!!!\n");
  }
  else {
    printf("zero=128 end: fail\n");
    test_pass = 0;
  }

  bench_free(B);
  bench_free(C);
}

int bench_act8_validate() {
  return (setting->checksum == 0x00000022) && test_pass;
}
