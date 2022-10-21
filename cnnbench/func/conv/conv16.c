#include <benchmark.h>
#include "cnn.h"

#define N N_SIZE
#define S 1

static uint16_t A[N][N];
static uint64_t vwidth;
static int test_pass;

void bench_conv16_prepare() {
  bench_srand(1);
  vwidth = 0x8888888888888888;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = bench_rand() & 0xffff;
    }
  }
  test_pass = 1;
}

void bench_conv16_run() {
  int k;              //kernel size
  int m;              //output size
  int32_t *B;        //cnn output
  int32_t *C;        //std output
  int8_t *kernel;     //kernel
  int pass;
  
  for (k=1; k<=5; k++) {
    m = (N - k) / S + 1;
    B = (int32_t *)bench_alloc(sizeof(int32_t) * m * m);
    C = (int32_t *)bench_alloc(sizeof(int32_t) * m * m);
    kernel = (int8_t *)bench_alloc(sizeof(int8_t) * k * k);
    pass = 1;

    /*kernel[0] = 10;
    kernel[1] = -10;
    kernel[2] = 10;
    kernel[3] = -10;
    kernel[4] = 10;
    kernel[5] = -10;
    kernel[6] = 10;
    kernel[7] = -10;
    kernel[8] = 10;*/
    printf("\nk=%d begin: \n", k);
    for (int i=0; i<k*k; i++) {
      kernel[i] = bench_rand() & 0xff;
      printf("  %d", kernel[i]);
    }
    printf("\n");

    LoadV_Width((uint64_t)&vwidth);

    int8_t *kernel_ptr = kernel;
    for (int i=0; i<k; i++) {
      LoadV_D_Kernel((uint64_t)(kernel_ptr), k, i, 0);
      kernel_ptr += k;
    }

    uint16_t *col_ptr;
    for (int i=0; i<m; i++) {
      col_ptr = &A[0][i];
      for (int l=0; l<k; l++) {
        LoadV_D_Main((uint64_t)(col_ptr), k, l, 0);
        col_ptr += N;
      }
      B[0 * m + i] = Conv(k);

      // std res
      int32_t tmp_res = 0;
      for (int si=0; si<k; si++) {
        for (int sj=0; sj<k; sj++) {
          tmp_res += A[0 + sj][i + si] * kernel[si * k + sj];
        }
      }
      C[0 * m + i] = tmp_res;//(tmp_res < 0) ? 0 : (tmp_res > 0xffff) ? 0xffff : tmp_res;

      if (B[0 * m + i] != C[0 * m + i]) {
        printf("  conv error: i=%d, j=0, conv_res=%d, std_res=%d, tmp_res=%d\n", i, B[0 * m + i], C[0 * m + i], tmp_res);
        for (int si=0; si<k; si++) {
          for (int sj=0; sj<k; sj++) {
            printf("  %d", A[0 + sj][i + si]);
          }
        }
        printf("\n");
        pass = 0;
      }
      else {
        printf("  ok: i=%d, j=0, std_res=%d, tmp_res=%d\n", i, B[0 * m + i], C[0 * m + i]);
      }

      for (int j=1; j<m; j++) {
        LoadV_P((uint64_t)(col_ptr), k, 0);
        B[j * m + i] = Conv(k);

        // std res
        int32_t tmp_res = 0;
        for (int si=0; si<k; si++) {
          for (int sj=0; sj<k; sj++) {
            tmp_res += A[j + sj][i + si] * kernel[si * k + sj];
          }
        }
        C[j * m + i] = tmp_res;//(tmp_res < 0) ? 0 : (tmp_res > 0xffff) ? 0xffff : tmp_res;

        if (B[j * m + i] != C[j * m + i]) {
          printf("  conv error: i=%d, j=%d, conv_res=%d, std_res=%d, tmp_res=%d\n", i, j, B[j * m + i], C[j * m + i], tmp_res);
          for (int si=0; si<k; si++) {
            for (int sj=0; sj<k; sj++) {
              printf("  %d", A[j + sj][i + si]);
            }
          }
          printf("\n");
          pass = 0;
        }
        else {
          printf("  ok: i=%d, j=%d, std_res=%d, tmp_res=%d\n", i, j, B[j * m + i], C[j * m + i]);
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
    bench_free(B);
    bench_free(C);
    bench_free(kernel);
  }
}

int bench_conv16_validate() {
  return (setting->checksum == 0x00000001) && test_pass;
}
