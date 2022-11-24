/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

//#include "../common/book.h"
#include <math.h>

#include "common/cpu_anim.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define THREAD 16

__global__ void kernel(unsigned char *ptr, int ticks) {
  // map from threadIdx/BlockIdx to pixel position
  int x = (DIM / THREAD) * threadIdx.x +
          blockIdx.x;  // oui = blockDim.x * (nbBlocks / nbThread)
  int y = (DIM / THREAD) * threadIdx.y +
          blockIdx.y;  // oui = blockDim.y * (nbBlocks / nbThread)
  int offset = x + y * DIM;
  float fx = x - DIM / 2;
  float fy = y - DIM / 2;
  float d = sqrtf(fx * fx + fy * fy);
  unsigned char grey =
      (unsigned char)(128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) /
                                   (d / 10.0f + 1.0f));
  ptr[offset * 4 + 0] = grey;
  ptr[offset * 4 + 1] = grey;
  ptr[offset * 4 + 2] = grey;
  ptr[offset * 4 + 3] = 255;
}

struct DataBlock {
  unsigned char *dev_bitmap;
  CPUAnimBitmap *bitmap;
};

void generate_frame(DataBlock *d, int ticks) {
  dim3 blocks(DIM / THREAD, DIM / THREAD);
  dim3 threads(THREAD, THREAD);
  kernel<<<blocks, threads>>>(d->dev_bitmap, ticks);
  cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(),
             cudaMemcpyDeviceToHost);
}

// clean up memory allocated on the GPU
void cleanup(DataBlock *d) {
  cudaFree(d->dev_bitmap);
  free(d->bitmap->get_ptr());
}

int main(void) {
  DataBlock data;
  CPUAnimBitmap bitmap(DIM, DIM, &data);
  data.bitmap = &bitmap;

  unsigned char *dev_bitmap;
  cudaMalloc(&dev_bitmap, bitmap.image_size());
  data.dev_bitmap = dev_bitmap;

  bitmap.anim_and_exit((void (*)(void *, int))generate_frame,
                       (void (*)(void *))cleanup);
}