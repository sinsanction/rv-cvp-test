#include "benchmark.h"
#include "cnnapi.h"
#include "cnnapi_common.h"
#include "cnnapi_stdins.h"

static image_mc_t *A;     // input
static image_mc_t *B;     // cnn output
static image_mc_t *C;     // std output
static kernel_mc_t *kernel;
static int w = 30;
static int h = 30;
static int k = 3;
static int strides = 2;
static int in_channel = 3;
static int out_channel = 2;
static out_scale_mc_t out_scale;

static int test_pass;

void bench_conv4_prepare() {
  bench_srand(1);
  A = RandomInitImage(w, h, 2, in_channel);
  kernel = RandomInitKernel(k, 1, in_channel, out_channel);
  out_scale.channel = out_channel;
  out_scale.scale = (out_scale_t *)malloc(sizeof(out_scale_t) * out_channel);
  for (int i=0; i<out_channel; i++) {
    out_scale.scale[i].scale = A->img[0]->scale / 2;
    out_scale.scale[i].zero_point = 1;
  }
  test_pass = 1;
  SetOutputKernel(kernel);
}

void bench_conv4_run() {
  B = Convolution(A, kernel, strides, &out_scale);
  C = StdIns_Convolution(A, kernel, strides, &out_scale);
}

int bench_conv4_validate() {
  if (B->width == C->width && B->height == C->height && B->channel == C->channel) {
    for (int c=0; c<B->channel; c++) {
      for (int j=0; j<B->width; j++) {
        for (int i=0; i<B->height; i++) {
          uint16_t temp1 = get_main_value((uint64_t *)(B->img[c]->addr), j * B->height + i, B->img[c]->vwidth);
          uint16_t temp2 = get_main_value((uint64_t *)(C->img[c]->addr), j * B->height + i, C->img[c]->vwidth);
          if (temp1 != temp2) {
            printf("  conv error: c=%d, i=%d, j=%d, conv_res=%d, std_res=%d\n", c, i, j, temp1, temp2);
            test_pass = 0;
          }
        }
      }
    }
  }
  else {
    test_pass = 0;
  }

  if (test_pass == 1) {
    printf("end: pass!!!\n");
  }
  else {
    printf("end: fail\n");
  }
  bench_free(A);
  bench_free(B);
  bench_free(C);
  bench_free(kernel);
  return (setting->checksum == 0x00000004) && test_pass;
}
