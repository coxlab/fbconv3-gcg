#raw
#include <stdio.h>
#end raw

#set NF=4
#set Z=4

texture<float4, 1, cudaReadModeElementType> tex_float4;
__constant__ float constant[$Z][$FILTER_W][$NF];

#define uint unsigned int
#define IMUL(a, b) __mul24(a, b)

extern "C" {

#for j in xrange($FILTER_H)

  __global__ void cudafilter_kernel_j${j}(float4 *input, float4 *output)
  {

    // if needed, pad smem to get maximum performance (avoid smem bank conflicts)
#if $INPUT_BLOCK_W % 16 == 0
    __shared__ float shared_in[$BLOCK_H][$Z][$INPUT_BLOCK_W];
#else
    __shared__ float shared_in[$BLOCK_H][$Z][$INPUT_BLOCK_W+1];
#end if

    // -- input/output offsets
/*     __shared__ uint s_in_idx[$BLOCK_H][$BLOCK_W+1], s_out_idx[$BLOCK_H][$BLOCK_W+1]; */
    const uint in_idx = blockIdx.y*$BLOCK_H*$INPUT_W + $j*$INPUT_W + threadIdx.y*$INPUT_W + blockIdx.x*$BLOCK_W + threadIdx.x;
    const uint out_idx = blockIdx.y*$BLOCK_H*$OUTPUT_W + threadIdx.y*$OUTPUT_W + blockIdx.x*$BLOCK_W + threadIdx.x;
/*     s_in_idx[threadIdx.y][threadIdx.x] = in_idx; */
/*     s_out_idx[threadIdx.y][threadIdx.x] = out_idx; */
    float4 input_v4;

    // -- load input to shared memory
#for i in xrange($N_LOAD_ITERATIONS)
#if $i==($N_LOAD_ITERATIONS-1)
    if((threadIdx.x+$BLOCK_W*$i)<$INPUT_BLOCK_W)
#end if
      {
	input_v4 = tex1Dfetch(tex_float4, in_idx+$BLOCK_W*$i);
	shared_in[threadIdx.y][0][threadIdx.x+$BLOCK_W*$i] = input_v4.x;
	shared_in[threadIdx.y][1][threadIdx.x+$BLOCK_W*$i] = input_v4.y;
	shared_in[threadIdx.y][2][threadIdx.x+$BLOCK_W*$i] = input_v4.z;
	shared_in[threadIdx.y][3][threadIdx.x+$BLOCK_W*$i] = input_v4.w;
      }
#end for
    __syncthreads();

    // -- compute dot products
    float v, w;

#for n in xrange($NF)
    float sum$n = 0;
#end for
    
#for d in xrange($Z)
#for i in xrange($FILTER_W)
    v = shared_in[threadIdx.y][$d][threadIdx.x+$i];
#for n in xrange($NF)
    w = constant[$d][$i][$n];
    sum$n += v*w;
#end for
#end for
#end for

/*     output[s_out_idx[threadIdx.y][threadIdx.x]].x += sum0; */
/*     output[s_out_idx[threadIdx.y][threadIdx.x]].y += sum1; */
/*     output[s_out_idx[threadIdx.y][threadIdx.x]].z += sum2; */
/*     output[s_out_idx[threadIdx.y][threadIdx.x]].w += sum3; */

    output[out_idx].x += sum0;
    output[out_idx].y += sum1;
    output[out_idx].z += sum2;
    output[out_idx].w += sum3;
  }  
#end for

}

