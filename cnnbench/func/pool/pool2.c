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

void bench_pool2_prepare() {
  bench_srand(1);
  vwidth = 0x1111111111111111;
  a_size = round_up_div(N * N, 4);
  A = (uint8_t *)bench_alloc(sizeof(uint8_t) * a_size);
  for (int i=0; i<a_size; i++) {
    A[i] = bench_rand() & 0xff;
  }
  test_pass = 1;
}

void bench_pool2_run() {
  int k;              //kernel size
  int m;              //output size
  uint8_t *Bm;        //pool max output
  uint8_t *Ba;        //pool avg output
  uint8_t *Cm;        //std max output
  uint8_t *Ca;        //std avg output
  int pass;
  
  for (k=1; k<=5; k++) {
    m = (N - k) / S + 1;
    Bm = (uint8_t *)bench_alloc(sizeof(uint8_t) * m * m);
    Ba = (uint8_t *)bench_alloc(sizeof(uint8_t) * m * m);
    Cm = (uint8_t *)bench_alloc(sizeof(uint8_t) * m * m);
    Ca = (uint8_t *)bench_alloc(sizeof(uint8_t) * m * m);
    pass = 1;

    printf("\nk=%d begin: \n", k);

    LoadV_Width((uint64_t)&vwidth);

    uint64_t col_ptr;
    for (int i=0; i<m; i++) {
      col_ptr = (((uint64_t)A) << 2) + i;
      for (int l=0; l<k; l++) {
        LoadV_D_Main((uint64_t)(col_ptr), k, l, 0);
        col_ptr += N;
      }
      Bm[0 * m + i] = Pool_Max(k);
      Ba[0 * m + i] = Pool_Avg(k);

      // std res
      uint8_t tmp_res_max = 0;
      uint32_t tmp_res_avg = 0;
      for (int si=0; si<k; si++) {
        for (int sj=0; sj<k; sj++) {
          tmp_res_max = max(tmp_res_max, get_main_uint2(A, (0 + sj) * N + (i + si)));
          tmp_res_avg += get_main_uint2(A, (0 + sj) * N + (i + si));
        }
      }
      Cm[0 * m + i] = tmp_res_max;
      uint32_t div = tmp_res_avg / (k * k);
      uint32_t rem = tmp_res_avg % (k * k);
      uint32_t cin = ((rem * 2) >= (k * k)) ? 1 : 0;
      Ca[0 * m + i] = div + cin;

      if ( (Bm[0 * m + i] != Cm[0 * m + i]) || (Ba[0 * m + i] != Ca[0 * m + i]) ) {
        printf("  pool error: i=%d, j=0, pool_max_res=%d, std_max_res=%d, pool_avg_res=%d, std_avg_res=%d, tmp_res_avg=%d\n", i, Bm[0 * m + i], Cm[0 * m + i], Ba[0 * m + i], Ca[0 * m + i], tmp_res_avg);
        for (int si=0; si<k; si++) {
          for (int sj=0; sj<k; sj++) {
            printf("  %d", get_main_uint2(A, (0 + sj) * N + (i + si)));
          }
        }
        printf("\n");
        pass = 0;
      }
      else {
        ;//printf("  ok: i=%d, j=0, std_max_res=%d, std_avg_res=%d, tmp_res_avg=%d\n", i, Bm[0 * m + i], Ba[0 * m + i], tmp_res_avg);
      }

      for (int j=1; j<m; j++) {
        LoadV_P((uint64_t)(col_ptr), k, 0);
        Bm[j * m + i] = Pool_Max(k);
        Ba[j * m + i] = Pool_Avg(k);

        // std res
        uint8_t tmp_res_max = 0;
        uint32_t tmp_res_avg = 0;
        for (int si=0; si<k; si++) {
          for (int sj=0; sj<k; sj++) {
            tmp_res_max = max(tmp_res_max, get_main_uint2(A, (j + sj) * N + (i + si)));
            tmp_res_avg += get_main_uint2(A, (j + sj) * N + (i + si));
          }
        }
        Cm[j * m + i] = tmp_res_max;
        uint32_t div = tmp_res_avg / (k * k);
        uint32_t rem = tmp_res_avg % (k * k);
        uint32_t cin = ((rem * 2) >= (k * k)) ? 1 : 0;
        Ca[j * m + i] = div + cin;

        if ( (Bm[j * m + i] != Cm[j * m + i]) || (Ba[j * m + i] != Ca[j * m + i]) ) {
          printf("  pool error: i=%d, j=%d, pool_max_res=%d, std_max_res=%d, pool_avg_res=%d, std_avg_res=%d, tmp_res_avg=%d\n", i, j, Bm[j * m + i], Cm[j * m + i], Ba[j * m + i], Ca[j * m + i], tmp_res_avg);
          for (int si=0; si<k; si++) {
            for (int sj=0; sj<k; sj++) {
              printf("  %d", get_main_uint2(A, (j + sj) * N + (i + si)));
            }
          }
          printf("\n");
          pass = 0;
        }
        else {
          ;//printf("  ok: i=%d, j=%d, std_max_res=%d, std_avg_res=%d, tmp_res_avg=%d\n", i, j, Bm[j * m + i], Ba[j * m + i], tmp_res_avg);
        }

        col_ptr += N;
      }
    }

    if (pass == 1) {
      printf("k=%d end: pass!!!\n", k);
    }
    else {
      printf("k=%d end: fail\n", k);
      test_pass = 0;
    }
    bench_free(Bm);
    bench_free(Ba);
    bench_free(Cm);
    bench_free(Ca);
  }
}

int bench_pool2_validate() {
  bench_free(A);
  return (setting->checksum == 0x00000014) && test_pass;
}
