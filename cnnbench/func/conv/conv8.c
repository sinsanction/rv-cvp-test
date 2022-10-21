#include <benchmark.h>
#include "cnn.h"

#define N N_SIZE
#define S 1

static uint8_t A[N][N];
static uint64_t vwidth;
static int test_pass;

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

inline int8_t get_kernel_int4(int8_t *kernel, int i) {
  int j = i / 2;
  int r = i % 2;
  int8_t kernel_i = (kernel[j] >> (r * 4)) & 0xf;
  if (kernel_i & 0x8) {
    kernel_i = kernel_i | 0xf0;
  }
  return kernel_i;
}

void bench_conv8_prepare() {
  bench_srand(1);
  vwidth = 0x4444444444444444;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = bench_rand() & 0xff;
    }
  }
  test_pass = 1;
}

void bench_conv8_run() {
  int k;              //kernel size
  int m;              //output size
  int32_t *B;         //cnn output
  int32_t *C;         //std output
  int8_t *kernel;     //kernel
  int k_size;
  int pass;
  
  for (k=1; k<=5; k++) {
    m = (N - k) / S + 1;
    B = (int32_t *)bench_alloc(sizeof(int32_t) * m * m);
    C = (int32_t *)bench_alloc(sizeof(int32_t) * m * m);
    k_size = round_up_div(k * k, 2);
    kernel = (int8_t *)bench_alloc(sizeof(int8_t) * k_size);
    pass = 1;

    printf("\nk=%d begin: \n", k);
    for (int i=0; i<k_size; i++) {
      kernel[i] = bench_rand() & 0xff;
    }
    printf("  k_size: %d\n", k_size);
    for (int i=0; i<k*k; i++) {
      printf("  %d", get_kernel_int4(kernel, i));
    }
    printf("\n");

    LoadV_Width((uint64_t)&vwidth);

    uint64_t kernel_ptr = ((uint64_t)kernel) << 1;
    for (int i=0; i<k; i++) {
      LoadV_D_Kernel(kernel_ptr, k, i, 0);
      kernel_ptr += k;
    }

    uint8_t *col_ptr;
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
          tmp_res += A[0 + sj][i + si] * get_kernel_int4(kernel, si * k + sj);
        }
      }
      C[0 * m + i] = tmp_res;//(tmp_res < 0) ? 0 : (tmp_res > 0xff) ? 0xff : tmp_res;

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
            tmp_res += A[j + sj][i + si] * get_kernel_int4(kernel, si * k + sj);
          }
        }
        C[j * m + i] = tmp_res;//(tmp_res < 0) ? 0 : (tmp_res > 0xff) ? 0xff : tmp_res;

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

int bench_conv8_validate() {
  return (setting->checksum == 0x00000002) && test_pass;
}
